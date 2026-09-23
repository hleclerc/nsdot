from .PowerDiagram import PowerDiagram


def Voronoi( positions, **kwargs ):
    """Le diagramme de Voronoï euclidien de `positions` : un `PowerDiagram` SANS poids.

    Ce n'est pas un cas particulier qu'on aurait câblé, c'est le même objet dit autrement. Dans le
    plan qui sépare deux germes, les poids n'entrent que par leur DIFFÉRENCE (voir
    `cell/Plane.h::bisector`) : des poids tous égaux ne déplacent aucun plan, donc « tous
    égaux » et « pas de poids » désignent le même diagramme. Autant ne rien porter -- `weights`
    reste `Unbound`, arrive en `NoneTensor` côté C++, et le terme de poids disparaît du kernel à la
    COMPILATION. Un `[ n ]` de zéros donnerait exactement le même résultat en le faisant lire.

    Une fonction et non une sous-classe : les sous-classes de `PowerDiagram` sont ses STOCKAGES
    (`PowerDiagram_Plain`, `PowerDiagram_Bsp`), chacune nommant sa structure C++, et un Voronoï
    n'est pas un stockage de plus. Tous les arguments de `PowerDiagram` passent, sauf `weights`
    -- en demander revient à demander un diagramme de puissance, et il porte déjà un nom.
    """
    if "weights" in kwargs:
        raise TypeError( "a Voronoi diagram carries no weights -- use `PowerDiagram` for that" )
    return PowerDiagram( positions, **kwargs )
