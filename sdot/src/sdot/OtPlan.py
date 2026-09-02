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

    = Cellules vides

    Une cellule peut devenir topologiquement VIDE en cours de descente (un poids trop bas devant
    ses voisins) -- son gradient, moindre-carrés comme barrière, y est alors identiquement nul (ou
    `NaN`, `1 / 0`, pour la barrière), un PLATEAU dont plus aucune itération ne sort. `_fit`
    combine donc deux choses, ni l'une ni l'autre suffisante seule : une répulsion DOUCE
    (`barrier_eps > 0`, un terme `sum_i masse_i / mesure_i` ajouté à l'objectif, qui grandit à
    l'approche du bord) et un rejet DUR dans la recherche linéaire (tout pas essayé qui laisserait
    une mesure sous `min_measure_fraction * min( masses )` est refusé sans même regarder
    l'objectif). Voir `_fit` pour le détail.

    Les positions et masses de `src_dist` sont lues UNE FOIS, comme des tableaux hôtes plutôt que
    des `Tensor` : ce sont les CONSTANTES de l'ajustement (on ne dérive que par rapport à `w`), et
    ça laisse `power_diagram` reconstruire un `PowerDiagram` frais, tracé sur `w` seul, à chaque
    pas -- exactement le montage déjà validé par `test_PowerDiagram::measures_derive_wrt_weights_alone`.
    """

    def __init__( self, src_dist, dst_dist, boundaries = None, accelerator = None,
                  max_nb_cuts = None, weights0 = None, max_iter = 200, ftol = 1e-13,
                  barrier_eps = 1e-4, min_measure_floor = 1e-6, memory = 10,
                  c1 = 1e-4, rho = 0.5, max_backtracks = 30, callback = None ):
        """`src_dist` : une `SumOfDiracs` (ses `weights`, normalisés, sont les masses cibles).
        `dst_dist` : la distribution CONTINUE contre laquelle intégrer (`Image`,
        `SumOfGaussians`, ...) -- normalisée à la même masse totale que `src_dist`.

        `boundaries` / `accelerator` / `max_nb_cuts` : transmis tels quels à chaque
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

        `callback( entry )`, s'il est donné, est appelé à CHAQUE pas accepté (`entry` un dict
        `step` / `weights` / `loss` / `min_measure` / `max_abs_residual`, la même chose que ce
        qui s'accumule dans `self.history`) -- pour qui veut suivre la descente sans relire
        l'historique après coup (un `print`, une barre de progression).
        """
        src_dist = src_dist.normalized_version()

        self.src_dist    = src_dist
        self.dst_dist    = dst_dist.normalized_version()
        self.boundaries  = boundaries
        self.accelerator = accelerator
        self.max_nb_cuts = max_nb_cuts

        d = int( src_dist.nb_dims.value )
        self._positions = np.asarray( src_dist.positions, dtype = float ).reshape( -1, d )
        self._masses    = np.asarray( src_dist.weights, dtype = float ).reshape( -1 )
        self._barrier_eps = float( barrier_eps ) if barrier_eps else 0.0

        w0 = ( np.zeros_like( self._masses ) if weights0 is None
             else np.asarray( weights0, dtype = float ).reshape( -1 ) )

        #: `( step, weights, loss, min_measure, max_abs_residual )` par pas ACCEPTÉ -- `step = 0`
        #: est le point de départ, avant le premier pas. De quoi tracer une courbe de convergence
        #: ou rejouer la descente (voir `sdot/tests/test_OtPlan.py::"ot 2D lbfgs"`) sans
        #: reconstruire quoi que ce soit : chaque entrée porte déjà les poids ET les diagnostics
        #: de son pas. `loss` est la perte AUGMENTÉE (moindre-carrés + barrière) : ce que `_fit`
        #: minimise effectivement -- `max_abs_residual` reste, lui, la seule chose qu'on cherche
        #: réellement à annuler.
        self.history = []

        self.weights = self._fit( w0, max_iter, ftol, min_measure_floor, memory,
                                  c1, rho, max_backtracks, callback )


    def power_diagram( self, weights = None ) -> PowerDiagram:
        """Le `PowerDiagram` pour `weights` (par défaut : les poids AJUSTÉS, `self.weights`)."""
        w = self.weights if weights is None else weights
        return PowerDiagram( self._positions, w, boundaries = self.boundaries,
                             accelerator = self.accelerator, max_nb_cuts = self.max_nb_cuts,
                             distribution = self.dst_dist )

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
        grad_fn = driver.jit( driver.grad( self._objective_raw ) )
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
