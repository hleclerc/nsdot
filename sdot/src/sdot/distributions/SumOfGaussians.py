from loom.tensor import Axis, CtShapeVar, RealTensor, ShapeVar
from loom.util import ComputedAttribute

from .Distribution import Distribution


class SumOfGaussians( Distribution ):
    """Une somme de gaussiennes ISOTROPES, comme densité continue.

        rho( x ) = somme_i  weights( i ) * exp( - |x - positions( i )|^2 / ( 2 sigmas( i )^2 ) )
                            / ( 2 pi sigmas( i )^2 ) ^ ( d / 2 )

    `weights( i )` est donc la MASSE de la gaussienne `i`, pas sa hauteur : la masse totale est la
    somme des poids, sans une intégrale à calculer, et normaliser n'est qu'une division.

    = Isotrope aujourd'hui, et comment on dira le contraire

    L'INTENTION se lit sur le RANG de `sigmas`, pas sur un drapeau :

        sigmas : RealTensor[ "num_gaussian" ]                   -- un scalaire  -> isotrope (ici)
        sigmas : RealTensor[ "num_gaussian", "dim" ]            -- un vecteur   -> alignée sur les axes
        sigmas : RealTensor[ "num_gaussian", "dim", "dim" ]     -- une matrice  -> quelconque

    C'est le bon endroit pour le dire parce que c'est là que la FFI le lit : le rang traverse dans le
    type C++ du membre, donc `SumOfGaussians::value_at` peut brancher dessus avec un
    `if constexpr ( sigmas.ct_rank == 1 )` -- pas un test à l'exécution, pas un second agrégat, et
    le cas isotrope ne paie rien pour l'existence des autres. Seul le rang 1 est écrit pour l'instant.

    = Ce qu'elle sert à tester

    C'est la première densité qui n'est PAS constante par morceaux, donc la première à faire passer
    l'intégrateur par sa quadrature (voir `PowerDiagram::integrate_into`). Elle ne découpe rien --
    son morceau est la cellule entière -- et ne fournit que trois choses ponctuelles : `value_at`,
    `gradient_at`, et où accumuler `d rho / d paramètres`. Elle ne sait rien des cellules, et
    `PowerDiagram` ne sait rien des gaussiennes : c'est exactement le partage qu'on veut éprouver.

    = Support non borné

    Contrairement à une image, elle ne donne pas de `bounding_half_spaces` : la tronquer perdrait de
    la masse, et pas dans le dos de l'appelant. Il faut donc lui donner un `box` (ou des
    `boundaries`) -- sans quoi les cellules du bord restent infinies et `measures` y répond
    `TF::max`, faute de simplices qui veuillent dire quelque chose. La masse hors du domaine est
    alors perdue, et la somme des mesures vaut la masse cible MOINS ces queues : c'est la vérité de
    ce qu'on a demandé, pas une erreur numérique.
    """

    nb_gaussians     : ShapeVar
    nb_dims          : CtShapeVar

    num_gaussian     : Axis[ "nb_gaussians" ]
    dim              : Axis[ "nb_dims" ]

    positions        : RealTensor[ "num_gaussian", "dim" ]
    sigmas           : RealTensor[ "num_gaussian" ]
    weights          : RealTensor[ "num_gaussian" ]

    current_mass     : ComputedAttribute[ RealTensor, ( "weights", ) ]

    def __init__( self, positions, sigmas, weights = None, target_mass = 1.0, **kwargs ):
        """`positions` : `[ n, d ]`. `sigmas` : `[ n ]` (isotrope). `weights` : `[ n ]`, la MASSE de
        chaque gaussienne -- toutes égales par défaut."""
        self.__base_init__( positions = positions, sigmas = sigmas, target_mass = target_mass, **kwargs )
        if weights is not None:
            self.weights = weights
        elif self.weights.is_undefined:
            # toutes de même masse : la valeur exacte n'a pas d'importance, `normalized_version` la
            # remet à l'échelle -- ce qui compte est qu'elles soient DÉFINIES, le kernel les lisant
            # sans branche (contrairement aux `weights` d'un `PowerDiagram`, où l'absence a un sens).
            self.weights = RealTensor[ *self.batch_axes, self.num_gaussian ].full( 1.0 )

    def normalized_version( self ):
        mass = self.mass
        if not self.target_mass.is_defined:
            return self

        # la masse totale est la somme des poids (chaque gaussienne est normalisée à `weights( i )`),
        # donc normaliser est une DIVISION et pas une intégrale -- et l'autodiff la traverse.
        return SumOfGaussians(
            nb_gaussians = self.nb_gaussians.value,
            nb_dims = self.nb_dims.value,

            positions = self.positions,
            sigmas = self.sigmas,
            weights = self.target_mass / mass * self.weights,

            current_mass = self.target_mass,
            batch_axes = self.batch_axes,
        )

    def _update_current_mass( self ):
        # réduction sur l'axe des GAUSSIENNES seulement, pour qu'un éventuel axe de batch survive
        # (même raison que `SumOfDiracs._update_current_mass`).
        self.current_mass = self.weights.sum( axis = self.num_gaussian )
