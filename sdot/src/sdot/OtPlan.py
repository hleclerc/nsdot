from collections import deque

import numpy as np

from loom.drivers.driver import driver

from .PowerDiagram import PowerDiagram


class OtPlan:
    """Plan de transport optimal semi-discret, en dimension quelconque : les poids d'un
    `PowerDiagram` tels que la mesure de chaque cellule contre `dst_dist` (une densité continue)
    égale la masse du dirac correspondant dans `src_dist` (une `SumOfDiracs`).

    `PowerDiagram.measures` fait déjà tout le travail géométrique et sait se DÉRIVER par rapport
    à `weights` -- il ne manque donc que la boucle qui les ajuste. Aujourd'hui : la moindre-carrés
    `0.5 * |mesure( w ) - masse|²` (`residual_loss`) minimisée par un L-BFGS. C'est un DÉTOUR par
    rapport à la vraie formulation du problème -- la fonctionnelle duale CONCAVE dont `mesure( w )
    - masse` est le GRADIENT (Kitagawa-Mérigot-Thibert) -- mais elle a le même point fixe : la
    jacobienne `∂mesure_i / ∂poids_j` est SYMÉTRIQUE (c'est le contenu géométrique du théorème de
    Brenier), donc le gradient d'ici, `J . résidu`, s'annule exactement là où `résidu` s'annule.
    Un Newton viendra ensuite EXPLOITER cette même jacobienne directement (elle est aussi la
    hessienne de la fonctionnelle duale) au lieu de la moindre-carrés -- voir `residual_loss`.
    En attendant, `objective = "dual"` descend DÉJÀ la fonctionnelle duale elle-même par L-BFGS
    ( `_fit_dual` ) : son gradient est le résidu tout court, valeur et gradient sortent d'un seul
    balayage ( `PowerDiagram.moments` ), et c'est ce qu'une reconstruction 3D utilise
    ( `otrec.models.ProjectedDiracModel` ) -- voir `__init__` pour ce qui distingue les deux.

    = Ce que le plan vaut

    `transport()` / `cost` / `cost_and_position_grad()` : le coût `W_2^2` aux poids ajustés, les
    barycentres des cellules, et la dérivée du coût par rapport aux POSITIONS des diracs par le
    théorème de l'enveloppe -- ce qu'un problème inverse ( les positions comme inconnues ) consomme.

    = Cellules vides

    Une cellule peut devenir topologiquement VIDE en cours de descente (un poids trop bas devant
    ses voisins) -- son gradient, moindre-carrés comme barrière, y est alors identiquement nul (ou
    `NaN`, `1 / 0`, pour la barrière), un PLATEAU dont plus aucune itération ne sort. `_fit`
    combine donc deux choses, ni l'une ni l'autre suffisante seule : une répulsion DOUCE
    (`barrier_eps > 0`, un terme `sum_i masse_i / mesure_i` ajouté à l'objectif, qui grandit à
    l'approche du bord) et un rejet DUR dans la recherche linéaire (tout pas essayé qui laisserait
    une mesure sous `min_measure_fraction * min( masses )` est refusé sans même regarder
    l'objectif). Voir `_fit` pour le détail.

    = Un seul diagramme

    Les positions sont les CONSTANTES de l'ajustement : le `PowerDiagram` est bâti UNE FOIS ( son
    arbre avec ), et chaque évaluation ne fait que lui POSER les poids essayés ( `pd.weights = w` )
    -- ce qui, pour un stockage BSP, refait le majorant des poids de chaque nœud et rien d'autre.
    Sous `driver.grad` / `jit`, les poids posés sont des traceurs, et ce qu'ils ont produit dans le
    diagramme ( poids triés, majorants ) l'est aussi une fois la trace close : c'est sans
    conséquence parce que TOUTE lecture du diagramme repasse par `power_diagram( w )`, qui pose des
    poids avant de rendre quoi que ce soit.
    """

    def __init__( self, src_dist, dst_dist, boundaries = None, accelerator = None,
                  kernel_dtype = None, weights0 = None, max_iter = 200, ftol = 1e-13,
                  barrier_eps = 1e-4, min_measure_floor = 1e-6, memory = 10,
                  c1 = 1e-4, rho = 0.5, max_backtracks = 30, callback = None, jit = True,
                  objective = "least_squares", mass_tol = 1e-8, damping = "kmt" ):
        """`src_dist` : une `SumOfDiracs` (ses `weights`, normalisés, sont les masses cibles).
        `dst_dist` : la distribution CONTINUE contre laquelle intégrer (`Image`,
        `SumOfGaussians`, ...) -- normalisée à la même masse totale que `src_dist`.

        `boundaries` / `accelerator` / `kernel_dtype` : transmis tels quels à chaque
        `PowerDiagram` construit pendant l'ajustement (voir `PowerDiagram.__init__`).

        `weights0` : le point de départ, par défaut `0` -- le diagramme de Voronoï, chaque
        cellule prenant sa part purement géométrique.

        `barrier_eps` : le poids du terme de répulsion (voir la docstring de la classe) -- `0`
        (ou `None`) le désactive, le rejet dur de `min_measure_floor` restant seul en jeu.
        `min_measure_floor` : la mesure la plus basse qu'une cellule ait le droit d'atteindre, en
        VALEUR ABSOLUE (l'unité est celle de `src_dist.weights` normalisés -- `1.0` de masse
        totale par défaut, voir `Distribution.normalized_version`). `_fit` la plafonne de toute
        façon à `0.1 *` la plus petite mesure DE DÉPART : sur un nuage déjà très déséquilibré au
        point de départ ( `weights0` ), un plancher fixe pourrait sinon interdire jusqu'au tout
        premier pas.

        `max_iter` / `ftol` : nombre de pas et écart de perte en dessous duquel on s'arrête.
        `memory` : nombre de paires `( s, y )` gardées par la récursion à deux boucles (voir
        `_two_loop_direction`). `c1` / `rho` / `max_backtracks` : la recherche linéaire par
        retour arrière (Armijo -- `c1` la pente minimale acceptée, `rho` le facteur de
        rétrécissement du pas, `max_backtracks` avant de basculer sur le secours en descente de
        gradient, voir `_fit`).

        `objective` : CE QUE la descente minimise --

        - `"least_squares"` ( défaut ) : `0.5 |mesure( w ) - masse|^2` + barrière, par L-BFGS sur son
          gradient `J . résidu` ( `_fit`, avec le plancher et la barrière décrits plus haut ) ;
        - `"dual"` : la fonctionnelle duale de Kantorovich elle-même, `-Phi( w )` avec
          `Phi( w ) = sum_i w_i ( nu_i - m_i( w ) ) + sum_i int_{cell_i} |x - p_i|^2 rho`, CONCAVE, dont
          le gradient est le RÉSIDU `m( w ) - nu` tout court ( `_fit_dual` ). Valeur et gradient
          sortent d'un seul balayage ( `PowerDiagram.moments` ), sans adjoint, et le
          conditionnement est celui de la jacobienne, pas son carré. Ni barrière ni plancher : une
          cellule vide a pour gradient sa masse cible, qui la fait revenir. Demande un domaine
          BORNÉ. `mass_tol` arrête la descente dès que `max |m - nu| <= mass_tol`. Sa précision est
          celle de la VALEUR de `Phi` : exacte sur une distribution constante par morceaux ( une
          `Image`, ou rien ), limitée à la quadrature ( `PointwiseDensity::rtol` ) sur une densité
          lisse -- où la moindre-carrés, qui n'a besoin que des mesures, va plus loin.

        - `"newton"` : la même fonctionnelle duale, par un NEWTON AMORTI ( `_fit_newton` ) : sa
          hessienne est la jacobienne des mesures par rapport aux poids, creuse, une entrée par
          facette ( `PowerDiagram.hessian_rows` ), et un système linéaire creux par pas. Le nombre
          de pas ne dépend plus du nombre de diracs -- c'est ce qu'il faut à un grand nuage, là où
          L-BFGS en demande de plus en plus. Mêmes limites que `"dual"` ( domaine borné, précision
          de la valeur ), plus : une distribution constante par morceaux ( `Image`, ou rien ).

        `damping` ( Newton seulement ) : `"kmt"` -- le pas est retenu dès que la norme du résidu
        baisse d'un facteur `1 - t/2` SANS qu'une cellule passe sous le plancher ( la moitié de la
        plus petite mesure de départ ), ce qui garantit la convergence à une vitesse bornée par ce
        plancher ; `"none"` -- le Newton NON amorti : le pas plein dès qu'il baisse le résidu, un
        retour arrière sinon, aucun plancher. Plus rapide loin de la solution quand rien n'est
        proche de zéro ( des projections floutées, `otrec` ), sans garantie sinon.

        Avant l'un comme l'autre, le point de départ est VÉRIFIÉ : si le Voronoï ( ou les poids
        donnés ) laisse une cellule vide -- des diracs hors du domaine -- les poids de départ sont
        ceux d'une SIMILITUDE qui ramène le nuage dans le domaine ( `_similarity_start` ) : le
        Voronoï du nuage translaté et contracté s'écrit comme un diagramme de puissance du nuage
        d'origine, `w_i = | p_i |^2 - | a p_i + b |^2 / a`, et ses cellules sont toutes nourries.

        `jit` : compiler le gradient de l'objectif une fois pour toute la descente ( voir `_fit` ).
        C'est ce qu'on veut pour UN ajustement long ; c'est ce qu'on ne veut pas pour BEAUCOUP de
        petits ajustements sur des nuages différents ( une reconstruction, `otrec` ), où chaque
        `OtPlan` recompilerait le sien -- `jit = False` évalue alors le gradient pas à pas.

        `callback( entry )`, s'il est donné, est appelé à CHAQUE pas accepté (`entry` un dict
        `step` / `weights` / `loss` / `min_measure` / `max_abs_residual`, la même chose que ce
        qui s'accumule dans `self.history`) -- pour qui veut suivre la descente sans relire
        l'historique après coup (un `print`, une barre de progression).
        """
        src_dist = src_dist.normalized_version()

        self.src_dist    = src_dist
        self.dst_dist    = dst_dist.normalized_version()

        d = int( src_dist.nb_dims.value )
        self._positions = np.asarray( src_dist.positions, dtype = float ).reshape( -1, d )
        self._masses    = np.asarray( src_dist.weights, dtype = float ).reshape( -1 )
        self._barrier_eps = float( barrier_eps ) if barrier_eps else 0.0
        self._jit = bool( jit )

        w0 = ( np.zeros_like( self._masses ) if weights0 is None
             else np.asarray( weights0, dtype = float ).reshape( -1 ) )

        # LE diagramme, bâti une fois sur les positions ( voir la docstring de la classe ) ; les
        # poids qu'il porte à un instant donné sont les derniers posés par `power_diagram( w )`
        self._pd = PowerDiagram( self._positions, w0, boundaries = boundaries, accelerator = accelerator,
                                 kernel_dtype = kernel_dtype, distribution = self.dst_dist )

        #: `( step, weights, loss, min_measure, max_abs_residual )` par pas ACCEPTÉ -- `step = 0`
        #: est le point de départ, avant le premier pas. De quoi tracer une courbe de convergence
        #: ou rejouer la descente (voir `sdot/tests/test_OtPlan.py::"ot 2D lbfgs"`) sans
        #: reconstruire quoi que ce soit : chaque entrée porte déjà les poids ET les diagnostics
        #: de son pas. `loss` est la perte AUGMENTÉE (moindre-carrés + barrière) : ce que `_fit`
        #: minimise effectivement -- `max_abs_residual` reste, lui, la seule chose qu'on cherche
        #: réellement à annuler.
        self.history = []

        if objective == "dual":
            self.weights = self._fit_dual( w0, max_iter, float( mass_tol ), memory, c1, rho, max_backtracks, callback )
        elif objective == "newton":
            self.weights = self._fit_newton( w0, max_iter, float( mass_tol ), rho, max_backtracks, callback, damping )
        elif objective == "least_squares":
            self.weights = self._fit( w0, max_iter, ftol, min_measure_floor, memory,
                                      c1, rho, max_backtracks, callback )
        else:
            raise ValueError( f"objective inconnu : { objective !r } ( 'least_squares' ou 'dual' )" )


    def power_diagram( self, weights = None ) -> PowerDiagram:
        """Le `PowerDiagram` pour `weights` (par défaut : les poids AJUSTÉS, `self.weights`) --
        toujours le MÊME objet, auquel on pose ces poids-là."""
        self._pd.weights = self.weights if weights is None else weights
        return self._pd

    def residual( self, weights = None ):
        """`mesure_i( weights ) - masse_i` -- ZÉRO au point cherché, DÉRIVABLE par rapport à
        `weights` (voir `PowerDiagram.measures`)."""
        w = self.weights if weights is None else weights
        return self.power_diagram( w ).measures - self._masses

    def residual_loss( self, weights = None ):
        """`0.5 * |résidu|²` -- la moindre-carrés SEULE, sans le terme de barrière (voir
        `_objective_raw` pour ce que `_fit` minimise réellement)."""
        r = self.residual( weights )
        return 0.5 * ( r * r ).sum()

    # -- ce que le plan VAUT, une fois ajusté ------------------------------------------------------

    def transport( self ):
        """`( cost, barycenters, masses )` aux poids AJUSTÉS : le coût `W_2^2 = sum_i int_{cell_i}
        |x - p_i|^2 rho` ( flottant ), le barycentre de chaque cellule ( `[ n, d ]` ) et sa masse
        ( `[ n ]` ), en tableaux hôtes -- lus sur les moments des cellules ( `PowerDiagram.moments` ).
        Une cellule vide garde son germe pour barycentre."""
        pd = self.power_diagram()
        mass, first, second = pd.moments
        m = np.asarray( mass ).reshape( -1 )
        mx = np.asarray( first ).reshape( len( m ), -1 )
        m2 = np.asarray( second ).reshape( -1 )
        p = self._positions
        cost = float( m2.sum() - 2 * ( p * mx ).sum() + ( m * ( p * p ).sum( axis = 1 ) ).sum() )
        safe = np.where( m > 0, m, 1.0 )[ :, None ]
        bary = np.where( m[ :, None ] > 0, mx / safe, p )
        return cost, bary, m

    @property
    def cost( self ):
        """Le coût de transport `W_2^2` entre les diracs et `dst_dist`, aux poids ajustés."""
        return self.transport()[ 0 ]

    def cost_and_position_grad( self ):
        """`( cost, grad )` : le coût, et sa dérivée par rapport aux POSITIONS des diracs
        ( `[ n, d ]` ) -- par le théorème de l'enveloppe : aux poids optimaux, la dérivée du coût
        par rapport à `p_i` ne passe pas par les cellules, et vaut `2 m_i ( p_i - b_i )`, `b_i` le
        barycentre de la cellule et `m_i` sa masse ( qui est la masse cible du dirac ). C'est la
        même formule que `OtPlan1d`, et ce qu'une reconstruction consomme ( `otrec` )."""
        cost, bary, m = self.transport()
        return cost, 2 * m[ :, None ] * ( self._positions - bary )

    @property
    def cell_masses( self ):
        """La mesure de chaque cellule, aux poids AJUSTÉS -- proche de `src_dist.weights` si
        l'ajustement a convergé (voir `residual`)."""
        return self.power_diagram().measures


    # -- l'objectif (moindre-carrés + barrière) ----------------------------------------------------

    def _evaluate( self, weights ):
        """`( mesures, perte augmentée )`, en tableaux/nombres HÔTES -- calcul EAGER, SANS
        dérivée : c'est ce dont la recherche linéaire a besoin à CHAQUE pas essayé (une valeur
        scalaire et un fait géométrique, `mesures.min()`), le gradient n'étant recalculé, lui,
        qu'une fois par pas RETENU (voir `_fit`). Les deux formules (ici et `_objective_raw`)
        DOIVENT rester la même arithmétique -- l'une sur des tableaux nus pour la vitesse, l'autre
        sur des `Tensor` pour la dérivée -- sans quoi Armijo compare des choses différentes."""
        m = np.asarray( self.power_diagram( weights ).measures ).reshape( -1 )
        r = m - self._masses
        loss = 0.5 * float( np.dot( r, r ) )
        if self._barrier_eps:
            loss += self._barrier_eps * float( np.sum( self._masses / np.maximum( m, 1e-300 ) ) )
        return m, loss

    def _objective_raw( self, weights ):
        """LA MÊME perte augmentée que `_evaluate`, mais en `Tensor` -- DÉRIVABLE (voir
        `driver.grad` dans `_fit`). `.tensor` en sortie : `driver.grad`/`driver.jit` veulent un
        tableau brut du backend, pas l'objet `Tensor` (voir `otrec.Reconstruction.scalar_loss`,
        même convention).

        Le terme de barrière s'écrit `( 1.0 / m ) * self._masses`, PAS `self._masses / m` : un
        `numpy.ndarray` À GAUCHE d'un opérateur essaie de convertir l'AUTRE côté en tableau avant
        de lui laisser sa chance (`Tensor.__array__`), ce qui échoue sous trace (`m` y est un
        tracer, pas une valeur -- `TracerArrayConversionError`, même symptôme que le bug
        `PowerDiagram` + boîte + `jit`, mais ici sous `driver.grad` seul et dans CE code-ci,
        pas dans `PowerDiagram`). Un `Tensor` toujours à GAUCHE de l'opérateur (`1.0 / m`, un
        flottant Python ne dispute jamais la priorité) passe par SA propre surcharge et l'évite.
        """
        m = self.power_diagram( weights ).measures
        r = m - self._masses
        loss = 0.5 * ( r * r ).sum()
        if self._barrier_eps:
            loss = loss + self._barrier_eps * ( ( 1.0 / m ) * self._masses ).sum()
        return loss.tensor


    # -- L-BFGS maison + recherche linéaire sous contrainte de faisabilité --------------------------

    def _fit( self, w0, max_iter, ftol, min_measure_floor, memory, c1, rho, max_backtracks,
             callback ):
        """L-BFGS maison (récursion à deux boucles, Nocedal & Wright algorithme 7.4) + recherche
        linéaire par retour arrière -- PAS `scipy.optimize.minimize` : sa recherche linéaire ne
        connaît que la valeur SCALAIRE de l'objectif, alors que ce qu'il faut rejeter ici est un
        fait GÉOMÉTRIQUE (une cellule vidée), pas seulement une valeur qui remonte -- voir la
        docstring de la classe. `floor` (au plus `min_measure_floor`, voir `__init__`, PLAFONNÉ à
        `0.1 *` la plus petite mesure de départ) : tout pas essayé qui laisserait UNE SEULE
        cellule en dessous est refusé SANS MÊME regarder l'objectif, retour arrière (`x *= rho`)
        et nouvel essai.

        Le pas EFFECTIVEMENT pris (LBFGS, ou le secours en descente de gradient si le retour
        arrière s'épuise -- garanti descendant pour un pas assez petit) alimente ensuite la
        mémoire de courbure, jamais le pas candidat rejeté -- même principe que
        `otrec.optimizers.SubspaceNewtonLBFGS`.

        Le gradient est JITÉ une fois (`grad_fn`, ci-dessous) et réutilisé à chaque pas : `x`
        garde la même forme du début à la fin, donc une seule trace/compilation sert toute la
        descente au lieu d'en refaire une par pas. Deux bugs l'interdisaient jusqu'ici -- un
        domaine en BOÎTE illisible sous `jit` (`box_min` / `box_max`, voir `JaxDriver.array`) et
        le terme de barrière de `_objective_raw` ci-dessus -- tous deux corrigés désormais.
        """
        s_hist, y_hist = deque( maxlen = memory ), deque( maxlen = memory )

        def record( step, w, m, loss ):
            entry = { "step": step, "weights": np.asarray( w ).copy(), "loss": loss,
                      "min_measure": float( m.min() ),
                      "max_abs_residual": float( np.max( np.abs( m - self._masses ) ) ) }
            self.history.append( entry )
            if callback is not None:
                callback( entry )
            return entry

        x = np.asarray( w0, dtype = float ).copy()
        m, f = self._evaluate( x )
        record( 0, x, m, f )
        if m.min() <= 0:
            # PAS un cas que le retour arrière puisse réparer : une cellule déjà VIDE aux poids de
            # départ a un gradient identiquement nul (voir la docstring de la classe), donc AUCUN
            # pas, si petit soit-il, ne la fait bouger -- un plancher positif serait alors
            # insatisfiable pour toujours (`floor = 0.1 * 0 = 0`, et `mesure > 0` resterait faux
            # indéfiniment). Seul un autre point de départ (`weights0`, par défaut `0` -- le
            # Voronoï, toujours non vide pour un germe intérieur) peut en sortir.
            raise ValueError( "OtPlan: a cell is already empty at the starting weights "
                              "(min measure = 0) -- try a milder weights0" )

        # PLAFONNÉ à `0.1 *` la plus petite mesure DE DÉPART : un nuage déjà déséquilibré au point
        # de départ ( `weights0` ) ne doit pas rendre le tout premier pas impraticable -- le
        # plancher protège contre la CHUTE à zéro, pas contre un déséquilibre déjà là.
        floor = min( float( min_measure_floor ), 0.1 * float( m.min() ) )

        # jité UNE FOIS : `x` garde la même forme tout du long, donc la trace/compilation d'ici
        # sert tous les pas au lieu d'en refaire une par pas (voir la docstring de la méthode).
        grad_fn = driver.grad( self._objective_raw )
        if self._jit:
            grad_fn = driver.jit( grad_fn )
        g = np.asarray( grad_fn( x ) )

        for it in range( 1, max_iter + 1 ):
            direction = _two_loop_direction( g, s_hist, y_hist )
            directional_deriv = float( np.dot( g, direction ) )
            if directional_deriv >= 0:                     # mémoire dégénérée (rare) : repli sur
                direction = -g                              # la descente de gradient pure.
                directional_deriv = float( np.dot( g, direction ) )

            x_new, m_new, f_new = self._backtrack( x, direction, f, directional_deriv,
                                                   floor, c1, rho, max_backtracks )
            if x_new is None:
                # le secours -- petit pas le long de `-g`, garanti descendant pour `t` assez petit,
                # et lui aussi retenu tant qu'il viderait une cellule.
                x_new, m_new, f_new = self._backtrack( x, -g, f, -float( np.dot( g, g ) ),
                                                       floor, c1, rho, max_backtracks )
            if x_new is None:
                break                                       # plus aucun pas praticable

            g_new = np.asarray( grad_fn( x_new ) )
            s, y = x_new - x, g_new - g
            sy = float( np.dot( s, y ) )
            if sy > 1e-12 * float( np.dot( s, s ) ):         # condition de courbure -- sans elle
                s_hist.append( s ); y_hist.append( y )       # l'approximation cesserait d'être PSD

            record( it, x_new, m_new, f_new )
            converged = abs( f - f_new ) < ftol
            x, m, f, g = x_new, m_new, f_new, g_new
            if converged:
                break

        return x

    # -- la fonctionnelle duale ---------------------------------------------------------------------

    def _dual( self, weights ):
        """`( -Phi( w ), m( w ) - nu, m( w ) )` : la valeur à minimiser, son gradient et les mesures,
        d'UN balayage ( voir `objective = "dual"` dans `__init__` )."""
        mass, first, second = self.power_diagram( weights ).moments
        m = np.asarray( mass ).reshape( -1 )
        mx = np.asarray( first ).reshape( len( m ), -1 )
        m2 = np.asarray( second ).reshape( -1 )
        p = self._positions
        transport = float( m2.sum() - 2 * ( p * mx ).sum() + ( m * ( p * p ).sum( axis = 1 ) ).sum() )
        phi = float( np.dot( weights, self._masses - m ) ) + transport
        return - phi, m - self._masses, m

    def _fit_dual( self, w0, max_iter, mass_tol, memory, c1, rho, max_backtracks, callback ):
        """L-BFGS ( même récursion que `_fit` ) sur `-Phi`, avec un retour arrière d'Armijo : chaque
        essai de pas donne AUSSI le gradient, donc le pas retenu n'en recalcule rien."""
        s_hist, y_hist = deque( maxlen = memory ), deque( maxlen = memory )

        def record( step, w, m, f ):
            entry = { "step": step, "weights": np.asarray( w ).copy(), "loss": f,
                      "min_measure": float( m.min() ),
                      "max_abs_residual": float( np.max( np.abs( m - self._masses ) ) ) }
            self.history.append( entry )
            if callback is not None:
                callback( entry )
            return entry

        x = np.asarray( w0, dtype = float ).copy()
        f, g, m = self._dual( x )
        record( 0, x, m, f )

        for it in range( 1, max_iter + 1 ):
            if np.abs( g ).max() <= mass_tol:
                break
            direction = _two_loop_direction( g, s_hist, y_hist )
            dd = float( np.dot( g, direction ) )
            if dd >= 0:
                direction, dd = -g, -float( np.dot( g, g ) )

            # Armijo, et à défaut la SIMPLE décroissance : la valeur d'une densité lisse n'est connue
            # qu'à la précision de sa quadrature ( `PointwiseDensity::rtol` ), et près de la solution
            # ce bruit dépasse la décroissance qu'Armijo exige -- un pas qui descend quand même reste
            # un progrès, et la descente s'arrête d'elle-même quand plus aucun ne descend.
            t, accepted, best = 1.0, None, None
            for _ in range( max_backtracks ):
                x_try = x + t * direction
                f_try, g_try, m_try = self._dual( x_try )
                if np.isfinite( f_try ):
                    if f_try <= f + c1 * t * dd:
                        accepted = ( x_try, f_try, g_try, m_try )
                        break
                    if f_try < f and ( best is None or f_try < best[ 1 ] ):
                        best = ( x_try, f_try, g_try, m_try )
                t *= rho
            accepted = accepted or best
            if accepted is None:
                break
            x_new, f_new, g_new, m_new = accepted

            s, y = x_new - x, g_new - g
            sy = float( np.dot( s, y ) )
            if sy > 1e-12 * float( np.dot( s, s ) ):
                s_hist.append( s ); y_hist.append( y )

            record( it, x_new, m_new, f_new )
            x, f, g, m = x_new, f_new, g_new, m_new

        return x

    def _hessian( self, weights ):
        """La jacobienne `d m / d w` aux poids `weights`, creuse ( `scipy.sparse.csr_matrix` ) --
        symétrique, semi-définie positive, de noyau les constantes ( ajouter le même nombre à tous
        les poids ne déplace aucun plan ). Voir `PowerDiagram.hessian_rows`."""
        import scipy.sparse as sp
        counts, ids, vals = self.power_diagram( weights ).hessian_rows()
        n = len( counts )
        rows = np.repeat( np.arange( n ), ids.shape[ 1 ] ).reshape( n, -1 )
        keep = ( np.arange( ids.shape[ 1 ] )[ None, : ] < counts[ :, None ] ) & ( ids >= 0 )
        r, c, v = rows[ keep ], ids[ keep ], vals[ keep ]
        off = sp.coo_matrix( ( -v, ( r, c ) ), shape = ( n, n ) )
        diag = np.bincount( r, weights = v, minlength = n )
        return ( off + sp.diags( diag ) ).tocsr(), diag

    def _domain_box( self ):
        """`( lo, hi )` du pavé du domaine ( celui que `PowerDiagram` a lu sur les demi-espaces, donc
        aussi le support d'une image ), ou `None` s'il n'y en a pas"""
        pd = self._pd
        if not pd.box_min.is_defined:
            return None
        return np.asarray( pd.box_min ).reshape( -1 ), np.asarray( pd.box_max ).reshape( -1 )

    def _similarity_start( self, margin = 0.1 ):
        """Les poids du Voronoï d'une SIMILITUDE du nuage qui le loge dans le domaine ( voir
        `damping` dans `__init__` ) : la boîte du nuage est contractée ( jamais dilatée ) et
        translatée dans le pavé du domaine réduit d'une marge. Zéro ( le Voronoï lui-même ) sans
        pavé, ou si le nuage y est déjà."""
        box = self._domain_box()
        p = self._positions
        if box is None:
            return np.zeros( len( p ) )
        lo, hi = box
        p_lo, p_hi = p.min( axis = 0 ), p.max( axis = 0 )
        span_dom = ( hi - lo ) * ( 1 - 2 * margin )
        span_pts = np.maximum( p_hi - p_lo, 1e-300 )
        a = float( min( 1.0, ( span_dom / span_pts ).min() ) )
        # le centre du nuage contracté sur le centre du domaine
        b = ( lo + hi ) / 2 - a * ( p_lo + p_hi ) / 2
        q = a * p + b
        return ( p * p ).sum( axis = 1 ) - ( q * q ).sum( axis = 1 ) / a

    def _fit_newton( self, w0, max_iter, mass_tol, rho, max_backtracks, callback, damping ):
        """Newton sur `-Phi` ( voir `objective = "newton"` et `damping` ) : la direction résout
        `H d = -( m - nu )` ( `H` régularisée d'un `epsilon` sur sa diagonale, qui fixe la constante
        libre ). Une cellule VIDE a une ligne nulle : on lui donne la diagonale moyenne, ce qui en
        fait un pas de gradient à l'échelle des autres, le temps qu'elle revienne."""
        from scipy.sparse.linalg import spsolve
        import scipy.sparse as sp
        if damping not in ( "kmt", "none" ):
            raise ValueError( f"damping inconnu : { damping !r } ( 'kmt' ou 'none' )" )

        def record( step, w, m, f, t = 1.0, nb_evals = 1 ):
            entry = { "step": step, "weights": np.asarray( w ).copy(), "loss": f,
                      "min_measure": float( m.min() ),
                      "max_abs_residual": float( np.max( np.abs( m - self._masses ) ) ),
                      "t": t, "nb_evals": nb_evals }
            self.history.append( entry )
            if callback is not None:
                callback( entry )
            return entry

        x = np.asarray( w0, dtype = float ).copy()
        f, g, m = self._dual( x )
        # un point de départ qui VIDE une cellule ( des poids hérités d'autres positions, voir
        # `otrec.models.ProjectedDiracModel` ) est pire que le Voronoï : la théorie de KMT part
        # d'un plancher strictement positif, et une cellule vide y revient en rampant ( son gradient
        # est constant, sa ligne de hessienne nulle ). On repart alors de zéro si c'est mieux.
        # ( essayé, et rejeté, deux façons de faire mieux que le Voronoï : ne « ranimer » que les
        # cellules vides, `w_i = max_j ( w_j - | p_i - p_j |^2 ) + h^2`, qui remet `p_i` dans sa
        # cellule sans toucher aux autres ; et un départ MULTI-ÉCHELLE, les poids d'un plan sur des
        # paquets de 16 diracs hérités par chacun. Les deux butent sur la même chose : les poids
        # d'une solution varient de centaines de `h^2` entre voisins ( un nuage de départ loin de
        # la cible ), donc les ranimés mangent leurs voisins et les frères d'un paquet se vident
        # les uns les autres -- 3000 cellules mortes sur 5000. Le Voronoï, lui, nourrit tout le
        # monde, et la phase linéaire de KMT ( ~50 pas ) est le prix de sa plus petite cellule. )
        if np.any( x != 0 ) and float( m.min() ) < 1e-3 * float( self._masses.min() ):
            x0 = np.zeros_like( x )
            f0, g0, m0 = self._dual( x0 )
            if float( m0.min() ) > float( m.min() ):
                x, f, g, m = x0, f0, g0, m0
        # ... et le Voronoï lui-même peut laisser des cellules VIDES ( des diracs hors du domaine ) :
        # alors la similitude qui ramène le nuage dedans ( voir `_similarity_start` )
        if float( m.min() ) <= 0:
            x1 = self._similarity_start()
            f1, g1, m1 = self._dual( x1 )
            if float( m1.min() ) > float( m.min() ):
                x, f, g, m = x1, f1, g1, m1
        record( 0, x, m, f )
        # le PLANCHER de Kitagawa-Mérigot-Thibert : aucun pas accepté ne laisse une cellule sous la
        # moitié de la plus petite mesure de départ ( ni de la plus petite masse cible ) -- c'est
        # ce qui garantit la convergence, à une vitesse qui dépend de ce plancher. Non amorti :
        # pas de plancher, et le pas d'essai repart toujours de 1.
        kmt = damping == "kmt"
        floor = 0.5 * min( float( m.min() ), float( self._masses.min() ) ) if kmt else -np.inf
        t_start = 1.0

        for it in range( 1, max_iter + 1 ):
            gn = float( np.linalg.norm( g ) )
            if np.abs( g ).max() <= mass_tol:
                break
            H, diag = self._hessian( x )
            alive = diag > 0
            if not alive.any():
                break
            mean_diag = float( diag[ alive ].mean() )
            fix = np.where( alive, 1e-8 * mean_diag, mean_diag )
            direction = spsolve( ( H + sp.diags( fix ) ).tocsc(), -g )
            if not np.isfinite( direction ).all():
                direction = -g / mean_diag
            # ( pas de région de confiance : essayé, borner le pas à `h^2` fait ramper la phase
            # linéaire loin de la solution -- 28 pas deviennent 60 sans converger )

            # l'amortissement de KMT : le pas `t` est retenu dès que le RÉSIDU a baissé d'autant
            # ( `|| g( t ) || <= ( 1 - t / 2 ) || g ||` ) sans passer sous le plancher ; à défaut,
            # le pas qui a le plus baissé le résidu
            # le pas d'essai repart de ( quatre fois ) celui qu'on vient d'accepter, pas de 1 :
            # dans la phase linéaire de KMT ( des cellules presque vides, `t ~ 1e-3` ), repartir
            # de 1 coûtait dix évaluations par pas pour retomber au même `t`. Quatre fois, pour
            # que la phase quadratique retrouve `t = 1` en quelques pas.
            t, accepted, best, nb_evals = t_start, None, None, 0
            for _ in range( max_backtracks ):
                x_try = x + t * direction
                f_try, g_try, m_try = self._dual( x_try )
                nb_evals += 1
                gn_try = float( np.linalg.norm( g_try ) )
                if np.isfinite( f_try ) and float( m_try.min() ) >= floor:
                    if gn_try <= ( ( 1 - t / 2 ) if kmt else 1.0 ) * gn:
                        accepted = ( x_try, f_try, g_try, m_try, t )
                        break
                    if gn_try < gn and ( best is None or gn_try < best[ 5 ] ):
                        best = ( x_try, f_try, g_try, m_try, t, gn_try )
                t *= rho
            accepted = accepted or ( best and best[ :5 ] )
            if accepted is None:
                break
            x, f, g, m, t = accepted
            t_start = min( 1.0, 4 * t ) if kmt else 1.0
            record( it, x, m, f, t, nb_evals )

        return x

    def _backtrack( self, x, direction, f, directional_deriv, floor, c1, rho, max_backtracks ):
        """Le premier `x + t * direction`, `t = 1, rho, rho², ...`, qui (a) laisse TOUTES les
        mesures au-dessus de `floor` et (b) satisfait Armijo. `( None, None, None )` si aucun des
        `max_backtracks` essais n'y arrive."""
        t = 1.0
        for _ in range( max_backtracks ):
            x_try = x + t * direction
            m_try, f_try = self._evaluate( x_try )
            if ( m_try.min() > floor and np.isfinite( f_try )
               and f_try <= f + c1 * t * directional_deriv ):
                return x_try, m_try, f_try
            t *= rho
        return None, None, None


def _two_loop_direction( g, s_hist, y_hist ):
    """La direction de descente L-BFGS, `-H_k g` -- `H_k` l'approximation de l'inverse de la
    hessienne construite par récursion à deux boucles sur les paires `( s, y )` gardées (les plus
    anciennes en tête, comme les rend un `deque`). `H_0 = gamma * I`, la mise à l'échelle usuelle
    (Nocedal & Wright, autour de leur algorithme 7.4) : sans elle le tout premier pas (mémoire
    vide, `gamma = 1`) serait `-g`, à l'échelle du GRADIENT plutôt que de l'objectif."""
    q = g.copy()
    alphas, rhos = [], []
    for s, y in zip( reversed( s_hist ), reversed( y_hist ) ):
        r = 1.0 / np.dot( y, s )
        a = r * np.dot( s, q )
        q -= a * y
        alphas.append( a )
        rhos.append( r )
    if s_hist:
        s_last, y_last = s_hist[ -1 ], y_hist[ -1 ]
        gamma = np.dot( s_last, y_last ) / np.dot( y_last, y_last )
    else:
        gamma = 1.0
    r = gamma * q
    for s, y, a, rho in zip( s_hist, y_hist, reversed( alphas ), reversed( rhos ) ):
        beta = rho * np.dot( y, r )
        r += s * ( a - beta )
    return -r
