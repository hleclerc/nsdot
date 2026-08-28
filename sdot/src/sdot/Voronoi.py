from .PowerDiagram import PowerDiagram


def Voronoi( positions, **kwargs ):
    """Le diagramme de Voronoï euclidien de `positions` : un `PowerDiagram` SANS poids.

    Ce n'est pas un cas particulier qu'on aurait câblé, c'est le même objet dit autrement. Dans le
    plan qui sépare deux germes, les poids n'entrent que par leur DIFFÉRENCE (voir
    `PowerDiagram.cxx::make_cell`) : des poids tous égaux ne déplacent aucun plan, donc « tous
    égaux » et « pas de poids » désignent le même diagramme. Autant ne rien porter -- `weights`
    reste `Unbound`, arrive en `NoneTensor` côté C++, et le terme de poids disparaît du kernel à la
    COMPILATION. Un `[ n ]` de zéros donnerait exactement le même résultat en le faisant lire.

    Une fonction et non une sous-classe : c'est le NOM de la classe Python qui nomme la structure
    C++ générée (`sdot/PowerDiagram.h`), donc une sous-classe demanderait un second en-tête pour
    ne rien y écrire de neuf. Tous les arguments de `PowerDiagram` passent, sauf `weights` -- en
    demander revient à demander un diagramme de puissance, et il porte déjà un nom.
    """
    if "weights" in kwargs:
        raise TypeError( "a Voronoi diagram carries no weights -- use `PowerDiagram` for that" )
    return PowerDiagram( positions, **kwargs )
