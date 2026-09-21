"""Le plan de transport semi-discret, en dimension quelconque : les poids d'un `PowerDiagram` tels
que la masse de chaque cellule contre `dst_dist` ( une densité continue ) égale la masse du dirac
correspondant dans `src_dist` ( une `SumOfDiracs` ).

TOUT L'AJUSTEMENT EST EN C++ ( `sdot/otplan/` ), en UN `driver.call` : le point de départ, le
Newton amorti, ses diagrammes, le laplacien, le solveur linéaire, l'amortissement. Python ne fait
que poser le problème et lire ce qui en sort. C'est ce que le banc `solvers_des_familles` a
conclu ( README § 3, § 7, § 9, § 10 ) :

  * le NEWTON AMORTI de Kitagawa-Mérigot-Thibert gagne partout, de 2x à 4x en diagrammes comme en
    temps, contre L-BFGS et le gradient conjugué sur le dual, quel que soit leur
    préconditionnement -- la difficulté du transport semi-discret n'est pas la non-linéarité du
    dual, c'est sa NON-RÉGULARITÉ ( les cellules qui se vident ), et un pas de gradient s'y
    écrase comme un pas de Newton, en moins bien ( `otplan/Newton.h` ) ;
  * la hessienne est le laplacien du graphe de Laguerre, assemblé SANS TRI depuis les facettes
    du diagramme qui a mesuré le résidu -- un balayage livre les deux ( `otplan/Balayage.h`,
    `otplan/Laplacien.h` ) ; Cholesky creux en 2D, multigrille algébrique au-delà et en grand
    ( `otplan/Lineaire.h` ) ;
  * le pas d'essai repart du dernier pas accepté, jamais de 1 ( dix diagrammes par pas gagnés
    dans la phase linéaire ) ; en 2D, les LIMITES des cellules qui s'écrasent le long de la
    direction remplacent les essais à l'aveugle ( `otplan/Limites.h`, quand il sera là ).

= Le point de départ

Newton demande un départ ADMISSIBLE ( aucune cellule vide ). Le Voronoï ( `weights0 = None` ) l'est
dès que les diracs sont dans le domaine ; sinon, ou si les poids donnés vident une cellule, le C++
choisit lui-même le meilleur des trois -- les poids donnés, le Voronoï, la SIMILITUDE qui ramène le
nuage dans le domaine -- et le dit ( `stats[ "depart" ]` ). Voir `otplan/Solve.h`.

= Ce que le plan vaut

`transport()` / `cost` / `cost_and_position_grad()` : le coût `W_2^2` aux poids ajustés, les
barycentres des cellules, et la dérivée du coût par rapport aux POSITIONS des diracs par le
théorème de l'enveloppe -- ce qu'un problème inverse ( les positions comme inconnues ) consomme.
Des `Tensor` de loom, sur le device du driver : rien ici ne passe par numpy.

= Un seul diagramme

Les positions sont les CONSTANTES de l'ajustement : le `PowerDiagram` est bâti UNE FOIS ( son arbre
avec ), et le solveur ne fait que lui POSER les poids essayés -- ce qui, pour un stockage BSP,
refait le majorant des poids de chaque nœud et rien d'autre. Ce qu'il écrit ( les poids triés, les
majorants ) sont des SORTIES de l'appel, reprises par le diagramme après coup : les entrées d'un
appel sont en lecture seule.
"""

import numpy as np                  # les seuls tableaux hôtes d'ici : les quelques plans du domaine

from loom.compilation.FfiCode import FfiCode
from loom.drivers.driver import driver
from loom.tensor import Axis, CtShapeVar, IntTensor, RealTensor, ShapeVar, Tensor
from loom.util import Aggregate

from .Cell import Cell
from .CellScratch import fp_size
from .PowerDiagram import PowerDiagram, axis_aligned_box
from .hull import supporting_half_spaces


# ce que `stats` porte, dans l'ordre de `otplan/Solve.h::Stat`
_STATS = [ "fin", "reste", "reste0", "nb_iter", "nb_diag", "nb_recul", "t_maj", "t_diag", "t_asm", "t_lin", "t_lim", "eps",
           "masse_domaine", "nb_deborde", "nb_cell_lim", "nb_tours_essai", "lin_nb_hier", "lin_nb_iter", "lin_pire", "depart", "t_total",
           "nb_etapes", "min_masse_depart" ]
# une ligne de `history`, dans l'ordre de `otplan/Solve.h::Hist`
_HISTORY = [ "step", "t", "residual_l2", "min_measure", "max_abs_residual", "nb_diag", "nb_evals", "s" ]
_FIN = { 0: "en cours", 1: "converge", 2: "max iterations", 3: "stagnation", 4: "solveur lineaire en echec" }
_DEPART = { 0: "weights0", 1: "voronoi", 2: "similitude" }
_LIN = { "auto": 0, "cholesky": 1, "amg": 2, "cg": 3 }
_PAS = { "trials": 0, "limits": 1 }
_CONTINUATION = { "never": 0, "auto": 1, "always": 2 }


class _Options( Aggregate ):
    """les réglages du solveur, tels que `otplan/Solve.h` les lit"""
    mass_tol       : RealTensor
    mass_rtol      : RealTensor
    t_min          : RealTensor
    mult_ok        : RealTensor
    facteur        : RealTensor
    beta0          : RealTensor
    mult_lim       : RealTensor
    confiance      : RealTensor
    conv_s0        : RealTensor
    conv_ratio     : RealTensor
    conv_min       : RealTensor
    conv_seuil     : RealTensor
    max_iter       : IntTensor
    max_reculs     : IntTensor
    lin            : IntTensor
    pas            : IntTensor
    trace          : IntTensor
    continuation   : IntTensor
    cap0           : IntTensor
    kernel_fp_size : CtShapeVar


class _History( Aggregate ):
    """un pas ACCEPTÉ par ligne ( `step = 0` : le départ ), et les poids de chaque pas si on les a
    demandés ( `weights`, sinon `Unbound` -- un tableau `[ pas, n ]` qu'on ne veut pas toujours )"""
    rows      : RealTensor[ "num_step", "num_hist" ]
    weights   : RealTensor[ "num_step", "num_point" ]
    num_step  : Axis[ "nb_steps" ]
    num_hist  : Axis[ "nb_hist" ]
    num_point : Axis[ "nb_points" ]
    nb_steps  : ShapeVar
    nb_hist   : ShapeVar
    nb_points : ShapeVar


class OtPlan:
    """voir la docstring du module"""

    def __init__( self, src_dist, dst_dist, boundaries = None, accelerator = None, kernel_dtype = None,
                  weights0 = None, max_iter = 100, mass_tol = 1e-8, mass_rtol = 0.0, step = "auto",
                  linear_solver = "auto", keep_weights = False, verbose = False, t_min = 1e-10,
                  max_backtracks = 60, restart_factor = 4.0, memory = None,
                  continuation = "auto", conv_start = None, conv_ratio = 2 ** 0.5, conv_min = None, conv_threshold = 1e-2,
                  domain_margin = 0.0 ):
        """`src_dist` : une `SumOfDiracs` ( ses `weights`, normalisés, sont les masses cibles ).
        `dst_dist` : la distribution CONTINUE contre laquelle intégrer ( `Image`, `SumOfGaussians`, ou `None` : la mesure de Lebesgue sur le domaine,
        ... ), normalisée -- puis remise à l'échelle de ce que le DOMAINE en contient ( voir
        `otplan/Solve.h` : ce qu'on résout est le transport vers la densité restreinte au domaine ).

        `boundaries` / `accelerator` / `memory` : transmis tels quels au `PowerDiagram` ( voir
        `PowerDiagram.__init__` ). Le domaine est le support de `dst_dist` ( une image ) intersecté
        avec `boundaries` ; s'il n'est PAS BORNÉ ( des gaussiennes sans `boundaries` ), il est
        complété par l'ENVELOPPE des diracs -- l'intersection de demi-espaces qui s'appuient sur le
        nuage ( `hull.supporting_half_spaces` ), écartés de `domain_margin` : chaque dirac est dedans,
        donc chaque cellule de Voronoï a une mesure positive, et le transport est celui vers la
        densité restreinte à ce domaine ( `stats[ "masse_domaine" ]` dit ce qu'il en contient ).

        `kernel_dtype` : le flottant dans lequel la géométrie se coupe -- `FP64` par défaut ICI, et
        non `FP32` comme pour un diagramme seul : le banc l'a mesuré ( README § 4 ), l'amortissement
        demande une décroissance stricte du résidu que le bruit d'une aire en `float` refuse bien
        avant la tolérance, et des germes proches perdent leurs chiffres dans les facettes.

        `weights0` : le point de départ, par défaut le Voronoï ( voir la docstring du module ).

        `max_iter` : pas de Newton, au plus. `mass_tol` : on s'arrête dès que `max_i | m_i - nu_i |
        <= mass_tol` ( ABSOLU, l'unité est celle des masses normalisées ) ; `mass_rtol` : ou dès que
        `max_i | m_i - nu_i | / nu_i <= mass_rtol` ( 0 : inactif ). `t_min` : le pas en dessous
        duquel on déclare la STAGNATION ; `max_backtracks` : divisions par deux du pas, au plus, par
        itération ; `restart_factor` : le prochain essai part de ce facteur fois le dernier pas
        accepté ( plafonné à 1 ).

        `step` : `"trials"` ( les essais de KMT ), `"limits"` ( 2D : les limites des cellules qui
        s'écrasent, quand la passe sera là ) ou `"auto"`. `linear_solver` : `"auto"`, `"cholesky"`,
        `"amg"`, `"cg"` ( voir `otplan/Lineaire.h` ).

        `continuation` : la CONTINUATION EN LARGEUR ( `otplan/Continuation.h` ) -- résoudre d'abord
        pour la densité convolée par une gaussienne large, puis de plus en plus étroite, chaque
        étape partant des poids de la précédente : ce qu'il faut à une densité qui se concentre
        ( des bosses étroites, des déserts où des cellules n'ont pas de masse, et où Newton direct
        stagne ). `"auto"` ( défaut ) la déclenche quand le départ laisse une cellule sous
        `conv_threshold` fois la plus petite masse cible ; `"always"` / `"never"`. `conv_start` : la
        première largeur ( défaut : la moitié du diamètre du domaine ) ; `conv_ratio` : le facteur
        entre deux largeurs ( `sqrt( 2 )`, le meilleur mesuré ) ; `conv_min` : la dernière avant la
        densité elle-même ( défaut : le quart de la plus petite largeur des gaussiennes, ou du pas
        de l'image ). Une image se convole sur sa grille, des gaussiennes par leurs largeurs.

        `keep_weights` : garder les poids de CHAQUE pas dans `history` ( pour rejouer la descente --
        un tableau `[ pas, n ]` ). `verbose` : la trace du C++, une ligne par pas.
        """
        # le solveur est du code HOTE sur la file CPU ( `otplan/Balayage.h` ) : un driver dont le device
        # est un GPU ne peut pas l'appeler aujourd'hui -- `SDOT_DEVICE=cpu`, ou un driver CPU
        if not driver.device.is_cpu:
            raise NotImplementedError( "OtPlan : le solveur tourne sur le CPU pour l'instant ( les balayages et le solveur "
                                       "lineaire sont du code hote ) ; choisir le device CPU ( SDOT_DEVICE=cpu )" )
        src_dist = src_dist.normalized_version()
        self.src_dist = src_dist
        #: la densité cible, normalisée -- ou `None` : la mesure de Lebesgue sur le domaine ( borné )
        self.dst_dist = None if dst_dist is None else dst_dist.normalized_version()

        d = int( src_dist.nb_dims.value )
        if kernel_dtype is None:
            kernel_dtype = "FP64"

        # le domaine, borné : le support de la densité et `boundaries` -- ou, s'ils ne bornent rien,
        # l'enveloppe des diracs ( voir `hull.py` )
        boundaries = self._bounded_domain( d, boundaries, float( domain_margin ) )

        # LE diagramme, bâti une fois sur les positions ( voir la docstring du module ) ; les poids
        # qu'il porte à un instant donné sont les derniers posés
        self._pd = PowerDiagram( src_dist.positions, RealTensor[ src_dist.num_dirac ].full( 0.0 ) if weights0 is None else weights0,
                                 boundaries = boundaries, accelerator = accelerator, kernel_dtype = kernel_dtype,
                                 distribution = self.dst_dist, memory = memory )
        pd = self._pd
        n = int( pd.nb_points.value )

        #: les masses cibles, indexées comme les cellules
        self._masses = RealTensor[ pd.num_point ]( src_dist.weights.raw )

        # le pas par les limites n'existe qu'en 2D ( `otplan/Limites.h` ) ; ailleurs, les essais
        step = { "auto": "limits" if d == 2 else "trials" }.get( step, step )
        if step == "limits" and d != 2:
            raise ValueError( "step = 'limits' : 2D seulement pour l'instant ( voir `otplan/Limites.h` )" )
        if step not in _PAS:
            raise ValueError( f"step inconnu : { step !r } ( 'auto', 'trials' ou 'limits' )" )
        if linear_solver not in _LIN:
            raise ValueError( f"linear_solver inconnu : { linear_solver !r } ( { ', '.join( _LIN ) } )" )
        if continuation not in _CONTINUATION:
            raise ValueError( f"continuation inconnue : { continuation !r } ( 'auto', 'always' ou 'never' )" )

        options = _Options(
            mass_tol = float( mass_tol ), mass_rtol = float( mass_rtol ), t_min = float( t_min ),
            mult_ok = float( restart_factor ), facteur = 0.9, beta0 = 0.25, mult_lim = 2.0, confiance = 0.0,
            conv_s0 = float( conv_start or 0.0 ), conv_ratio = float( conv_ratio ), conv_min = float( conv_min or 0.0 ),
            conv_seuil = float( conv_threshold ),
            max_iter = int( max_iter ), max_reculs = int( max_backtracks ), lin = _LIN[ linear_solver ],
            pas = _PAS[ step ], trace = int( bool( verbose ) ), continuation = _CONTINUATION[ continuation ],
            cap0 = int( pd._scratch_capacity ),
            kernel_fp_size = fp_size( pd.kernel_dtype ),
        )

        weights = RealTensor[ pd.num_point ]()
        history = _History( nb_hist = len( _HISTORY ), nb_points = n )
        stats = RealTensor[ Axis( ShapeVar( len( _STATS ) ), name = "num_stat" ) ]()
        w0 = RealTensor[ pd.num_point ]( pd.weights.raw )

        dom = pd._domain_cell()
        dist_expr, _, dist_kwargs = pd._dist_for()
        pd_expr, pd_kwargs, pd_produced = pd._solver_weights_call()

        out = [ "weights", "history", "stats" ] + pd_kwargs[ "output_attributes" ]
        driver.call(
            FfiCode( name = "otplan_solve",
                includes = [ "sdot/otplan/Solve.h" ],
                sources = [ "sdot/otplan/Lineaire.cpp" ],
                fwd_code = "\n".join( [
                    "using TK_otplan = std::conditional_t<CT_VALUE( options.kernel_fp_size ) == 64, double, float>;",
                    f"auto pd_otplan = { pd_expr };",
                    "otplan::OptionsSolveur os;",
                    "otplan::NewtonOptions &no = os.newton;",
                    "no.tol_abs = double( options.mass_tol ); no.tol_rel = double( options.mass_rtol ); no.t_min = double( options.t_min );",
                    "no.mult_ok = double( options.mult_ok ); no.facteur = double( options.facteur ); no.beta0 = double( options.beta0 );",
                    "no.mult_lim = double( options.mult_lim ); no.confiance = double( options.confiance );",
                    "no.maxit = int( SI( options.max_iter ) ); no.max_reculs = int( SI( options.max_reculs ) );",
                    "no.pas = int( SI( options.pas ) ); no.trace = SI( options.trace ) != 0;",
                    "os.lin = otplan::Lin( int( SI( options.lin ) ) ); os.cap0 = SI( options.cap0 );",
                    "os.continuation = int( SI( options.continuation ) ); os.seuil_continuation = double( options.conv_seuil );",
                    "os.conv_s0 = double( options.conv_s0 ); os.conv_ratio = double( options.conv_ratio ); os.conv_min = double( options.conv_min );",
                    f"otplan::resoudre<TK_otplan>( queue, pd_otplan, dom_cell, { dist_expr }, nu, w0, os, weights, history, stats );",
                ] ) ),
            output_attributes = out,
            output_exceptions = [] if keep_weights else [ "history.weights" ],
            output_capacities = { "history.nb_steps": int( max_iter ) + 1 },
            has_dynamic_capacity = False,
            power_diagram = pd,
            dom_cell = dom,
            nu = self._masses,
            w0 = w0,
            options = options,
            weights = weights,
            history = history,
            stats = stats,
            **pd_kwargs[ "args" ],
            **dist_kwargs,
        )
        pd._solver_weights_after( pd_produced )

        #: les poids AJUSTÉS, `[ n ]`, indexés comme `positions` ( `weights[ 0 ] == 0` : la jauge )
        self.weights = weights
        #: ce que le solveur rapporte ( voir `otplan/Solve.h::Stat` ), plus `fin` et `depart` en clair
        st = stats.raw
        self.stats = { name: float( st[ k ] ) for k, name in enumerate( _STATS ) }
        self.stats[ "fin" ] = _FIN.get( int( self.stats[ "fin" ] ), "?" )
        self.stats[ "depart" ] = _DEPART.get( int( self.stats[ "depart" ] ), "?" )
        for name in ( "nb_iter", "nb_diag", "nb_recul", "nb_deborde", "nb_cell_lim", "nb_tours_essai", "lin_nb_hier", "lin_nb_iter", "nb_etapes" ):
            self.stats[ name ] = int( self.stats[ name ] )

        #: un dict par pas ACCEPTÉ -- `step = 0` est le point de départ : `t`, `residual_l2`,
        #: `min_measure`, `max_abs_residual`, `nb_diag` ( cumulé ), `nb_evals` ( les diagrammes de
        #: ce pas ), et `weights` si `keep_weights`
        nb_steps = int( history.nb_steps.value )
        rows = history.rows.raw[ :nb_steps ]
        self.history = []
        for s in range( nb_steps ):
            entry = { name: float( rows[ s, k ] ) for k, name in enumerate( _HISTORY ) }
            entry[ "step" ] = int( entry[ "step" ] )
            if keep_weights:
                entry[ "weights" ] = RealTensor[ pd.num_point ]( history.weights.raw[ s ] )
            self.history.append( entry )

    def _bounded_domain( self, d, boundaries, margin ):
        """`boundaries`, complétées par l'enveloppe des diracs si, avec le support de la densité,
        elles ne bornent pas le domaine ( voir `__init__` )"""
        planes = [] if boundaries is None else [ ( boundaries[ 0 ], boundaries[ 1 ] ) ]
        support = None if self.dst_dist is None else self.dst_dist.bounding_half_spaces()
        if support is not None:
            planes.append( support )
        if planes:
            dirs = [ d_ for d_, _ in planes ]
            offs = [ o_ for _, o_ in planes ]
            all_dirs = np.concatenate( [ np.asarray( x, dtype = float ).reshape( -1, d ) for x in dirs ] )
            all_offs = np.concatenate( [ np.asarray( x, dtype = float ).reshape( -1 ) for x in offs ] )
            if axis_aligned_box( all_dirs, all_offs ) is not None:
                return boundaries                                    # un pavé : borné
            dom = Cell.make_unbounded( d, kernel_dtype = "FP64" )   # un polytope quelconque : on le construit
            for k in range( len( all_offs ) ):
                dom.cut( all_dirs[ k ], float( all_offs[ k ] ) )
            if dom.is_bounded:
                return boundaries
        hd, ho = supporting_half_spaces( self.src_dist.positions, margin = margin )
        if boundaries is None:
            return hd, ho
        return ( np.concatenate( [ np.asarray( boundaries[ 0 ], dtype = float ).reshape( -1, d ), hd ] ),
                 np.concatenate( [ np.asarray( boundaries[ 1 ], dtype = float ).reshape( -1 ), ho ] ) )

    @property
    def converged( self ):
        return self.stats[ "fin" ] == "converge"

    def power_diagram( self, weights = None ) -> PowerDiagram:
        """Le `PowerDiagram` pour `weights` ( par défaut : les poids AJUSTÉS, `self.weights` ) --
        toujours le MÊME objet, auquel on pose ces poids-là."""
        self._pd.weights = self.weights if weights is None else weights
        return self._pd

    @property
    def target_masses( self ) -> Tensor:
        """`nu` : la masse cible de chaque cellule ( les masses des diracs, normalisées, puis remises
        à l'échelle de ce que le domaine contient -- voir `otplan/Solve.h` )"""
        return self._masses * ( self.stats[ "masse_domaine" ] / float( self._masses.sum() ) )

    def residual( self, weights = None ) -> Tensor:
        """`mesure_i( weights ) - nu_i` -- ZÉRO au point cherché, DÉRIVABLE par rapport à `weights`
        ( voir `PowerDiagram.measures` )."""
        w = self.weights if weights is None else weights
        return self.power_diagram( w ).measures - self.target_masses

    # -- ce que le plan VAUT, une fois ajusté ------------------------------------------------------

    def transport( self ):
        """`( cost, barycenters, masses )` aux poids AJUSTÉS : le coût `W_2^2 = sum_i int_{cell_i}
        |x - p_i|^2 rho` ( un `Tensor` scalaire ), le barycentre de chaque cellule ( `[ n, d ]` ) et
        sa masse ( `[ n ]` ) -- lus sur les moments des cellules ( `PowerDiagram.moments` ). Une
        cellule vide garde son germe pour barycentre."""
        pd = self.power_diagram()
        mass, first, second = pd.moments
        p = pd.positions
        cost = second.sum() - 2 * ( p * first ).sum() + ( mass * ( p * p ).sum( pd.dim ) ).sum()
        alive = mass > 0
        safe = alive.where( mass, 1.0 )
        bary = alive.where( first / safe, p )
        return cost, bary, mass

    @property
    def cost( self ):
        """Le coût de transport `W_2^2` entre les diracs et `dst_dist`, aux poids ajustés ( un
        `Tensor` scalaire : `float( plan.cost )` pour le nombre )."""
        return self.transport()[ 0 ]

    def cost_and_position_grad( self ):
        """`( cost, grad )` : le coût, et sa dérivée par rapport aux POSITIONS des diracs
        ( `[ n, d ]` ) -- par le théorème de l'enveloppe : aux poids optimaux, la dérivée du coût
        par rapport à `p_i` ne passe pas par les cellules, et vaut `2 m_i ( p_i - b_i )`, `b_i` le
        barycentre de la cellule et `m_i` sa masse ( qui est la masse cible du dirac ). C'est la
        même formule que `OtPlan1d`, et ce qu'une reconstruction consomme ( `otrec` )."""
        cost, bary, m = self.transport()
        return cost, 2 * m * ( self._pd.positions - bary )

    @property
    def cell_masses( self ) -> Tensor:
        """La mesure de chaque cellule, aux poids AJUSTÉS -- proche de `target_masses` si
        l'ajustement a convergé ( voir `residual` )."""
        return self.power_diagram().measures
