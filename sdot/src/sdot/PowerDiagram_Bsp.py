"""Les germes dans l'ORDRE D'UN ARBRE BSP : `sorted_positions` / `sorted_weights` sont les germes
permutés comme `tree.seed_indices` les range, de sorte qu'une feuille se lit d'un seul tenant, et
chaque cellule n'est coupée que par les germes que l'arbre n'a pas su écarter ( `AaBsp.py`,
`cell/Fournisseurs.h::FournisseurBsp` ).

Tout ce que le kernel manipule est dans cet ordre-là -- les rangs, les identifiants de coupe, les
gradients sur les germes -- et c'est ici qu'on traduit : `positions` / `weights` rendent les germes
dans l'ordre de l'utilisateur, les mesures et les cellules sortent du kernel déjà à leur indice
( `user_id( k )` ), et le rassemblement `sorted = positions[ seed_indices ]` est une opération du
backend, DÉRIVABLE : une dérivée par rapport à `sorted_positions` revient sur `positions` toute seule.

Changer les POIDS ne change pas l'arbre, seulement le majorant affine que chaque nœud porte
( `refresh_weight_majorants` ) : c'est ce qui rend un diagramme réutilisable d'un pas à l'autre
d'un ajustement ( `OtPlan` ). Changer les POSITIONS le rebâtit.

LA MÉMOIRE ( `memo_nbrs [ n, K ]`, `memo_counts [ n ]`, en rangs de l'arbre ) : les voisins de
chaque cellule au dernier `measures`, que le fournisseur propose en premier au suivant
( `cell/Fournisseurs.h`, `MEMO` ). Écrite par le kernel de `measures` dans deux tenseurs neufs,
repris ici après l'appel ( les entrées et les sorties d'un appel sont disjointes ) -- sauf sous une
trace, où ce qui sort est un traceur : la mémoire d'avant reste, elle vaut toujours. Effacée avec
l'arbre, quand les positions changent. `memory = 0` ne la nomme pas : `NoneTensor` côté C++, et le
chemin ordinaire à la compilation.
"""

import numpy as np

from loom.drivers.driver import driver
from loom.tensor import Axis, IntTensor, RealTensor, ShapeVar

from .AaBsp import AaBsp
from .PowerDiagram import PowerDiagram


class PowerDiagram_Bsp( PowerDiagram ):
    tree             : AaBsp

    sorted_positions : RealTensor[ "num_point", "dim" ]
    sorted_weights   : RealTensor[ "num_point" ]

    memo_nbrs        : IntTensor[ "num_point", "num_memo", dict( size = 32 ) ]
    memo_counts      : IntTensor[ "num_point", dict( size = 32 ) ]
    num_memo         : Axis[ "nb_memo" ]
    nb_memo          : ShapeVar

    def _init_seeds( self, positions, weights, accelerator ):
        tree = accelerator if isinstance( accelerator, AaBsp ) else AaBsp( positions, weights )
        n = int( tree.nb_bsp_seeds.value )
        if n != int( positions.shape[ 0 ] ):
            raise ValueError( f"the accelerator was built on { n } seeds, this diagram has { int( positions.shape[ 0 ] ) }" )
        self._order = np.asarray( tree.seed_indices ).reshape( -1 ).astype( np.int64 )
        self._rank_of = tree.rank_of_seeds()
        res = { "tree": tree, "sorted_positions": self._gather( positions ) }
        if weights is not None:
            res[ "sorted_weights" ] = self._gather( weights )
            # un arbre venu de l'extérieur a pu être bâti sur d'autres poids, ou sans : son majorant
            # est refait sur CEUX-CI ( bâti ici, il les a déjà )
            if tree is accelerator:
                tree.refresh_weight_majorants( res[ "sorted_positions" ], res[ "sorted_weights" ] )
        # la mémoire part vide ( aucun souvenir : le chemin ordinaire, en attendant le premier `measures` )
        K = int( getattr( self, "_memory", 0 ) )
        res[ "nb_memo" ] = K
        if K > 0:
            res[ "memo_nbrs" ] = np.zeros( ( n, K ), dtype = np.int32 )
            res[ "memo_counts" ] = np.zeros( n, dtype = np.int32 )
        return res

    def _memo_for_call( self ):
        if not self.memo_counts.is_defined:
            return "0, 0", {}, None
        nbrs = IntTensor[ self.num_point, self.num_memo, dict( size = 32 ) ]()
        counts = IntTensor[ self.num_point, dict( size = 32 ) ]()
        return "memo_nbrs_out, memo_counts_out", dict(
            call = dict( output_attributes = [ "memo_nbrs_out", "memo_counts_out" ] ),
            args = dict( memo_nbrs_out = nbrs, memo_counts_out = counts ) ), ( nbrs, counts )

    def _memo_after_call( self, produced ):
        if produced is None:
            return
        nbrs, counts = produced
        if driver.is_traced( counts.raw ):          # sous une trace : on garde les souvenirs d'avant
            return
        self.memo_nbrs = nbrs.raw
        self.memo_counts = counts.raw

    def _gather( self, seeds ):
        """`seeds[ seed_indices ]`, par le backend : un traceur y reste un traceur, et la dérivée
        par rapport à ce qu'on rassemble repasse par ici toute seule"""
        raw = getattr( seeds, "raw", seeds )
        return raw[ self._order ]

    def _scatter( self, sorted_values ):
        """l'inverse : ce qui est rangé dans l'ordre de l'arbre, remis dans celui de l'utilisateur"""
        return sorted_values[ self._rank_of ]

    # ---- ce que l'utilisateur lit et écrit ---------------------------------------------------------

    @property
    def positions( self ):
        """`[ n, d ]`, dans l'ordre de l'utilisateur -- un tenseur NEUF, rassemblé depuis le stockage"""
        return RealTensor[ self.num_point, self.dim ]( self._scatter( self.sorted_positions.raw ) )

    @positions.setter
    def positions( self, positions ):
        """de nouvelles positions : l'arbre est rebâti dessus ( il faut donc des valeurs concrètes )"""
        pos = positions if hasattr( positions, "shape" ) else np.asarray( positions, dtype = float )
        w = self.sorted_weights.raw[ self._rank_of ] if self.sorted_weights.is_defined else None
        for name, value in self._init_seeds( pos, w, None ).items():
            setattr( self, name, value )
        self._dom_cell = None

    @property
    def weights( self ):
        """`[ n ]` dans l'ordre de l'utilisateur, ou un tenseur `Unbound` ( `is_defined == False` )
        quand le diagramme n'en porte pas -- la même chose que ce que `PowerDiagram_Plain` stocke"""
        res = RealTensor[ self.num_point ]()
        if self.sorted_weights.is_defined:
            res.set( self._scatter( self.sorted_weights.raw ) )
        return res

    @weights.setter
    def weights( self, weights ):
        """de nouveaux poids : les germes ne bougent pas, l'arbre non plus -- seul le majorant affine
        des poids de chaque nœud est refait, en un kernel sur les nœuds"""
        self.sorted_weights = self._gather( weights )
        self.tree.refresh_weight_majorants( self.sorted_positions, self.sorted_weights )

    def _ranks_of_items( self ):
        return self._rank_of

    # ---- ce que le solveur de `OtPlan` ecrit ---------------------------------------------------------

    def _solver_weights_call( self ):
        """`( expression C++ du diagramme aux poids INSCRIPTIBLES, kwargs de l'appel, ce qu'il faut
        reprendre après )` : les poids triés et les majorants de l'arbre sont des SORTIES de l'appel
        ( `PowerDiagram_Bsp.h::with_weights` ), que `_solver_weights_after` adopte"""
        sw = RealTensor[ self.num_point ]()
        wa = RealTensor[ self.tree.num_bsp_node, self.tree.dim ]()
        wb = RealTensor[ self.tree.num_bsp_node ]()
        args = dict( sorted_weights_out = sw, node_wa_out = wa, node_wb_out = wb )
        expr = "power_diagram.with_weights( sorted_weights_out, node_wa_out, node_wb_out"
        # la memoire aussi ( `memory > 0` ) : le solveur la refait a chaque balayage, dans deux sorties
        # neuves qu'il initialise depuis les souvenirs d'avant
        if self.memo_counts.is_defined:
            nbrs = IntTensor[ self.num_point, self.num_memo, dict( size = 32 ) ]()
            counts = IntTensor[ self.num_point, dict( size = 32 ) ]()
            args.update( memo_nbrs_out = nbrs, memo_counts_out = counts )
            expr += ", memo_nbrs_out, memo_counts_out"
        return ( expr + " )", dict( output_attributes = list( args ), args = args ), args )

    def _solver_weights_after( self, produced ):
        self.sorted_weights = produced[ "sorted_weights_out" ].raw
        self.tree.node_wa = produced[ "node_wa_out" ].raw
        self.tree.node_wb = produced[ "node_wb_out" ].raw
        if "memo_counts_out" in produced:
            self.memo_nbrs = produced[ "memo_nbrs_out" ].raw
            self.memo_counts = produced[ "memo_counts_out" ].raw

    def _grad_seeds_expr( self ):
        return "grad_for_power_diagram.sorted_positions, grad_for_power_diagram.sorted_weights"
