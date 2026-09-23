from loom.util import Aggregate


class SpatialAccelerator( Aggregate ):
    """Ce qui répond à « QUELS germes valent la peine d'être essayés, et dans quel ordre ».

    Un accélérateur ne sait pas ce qu'est une cellule. Il connaît la répartition des germes dans
    l'espace, et il en tire une ÉNUMÉRATION : au lieu des `n - 1` bissectrices que
    `PowerDiagram_Plain` essaie une par une, il propose les germes proches d'abord et s'arrête
    d'explorer une région dès que l'appelant lui dit qu'elle ne peut plus rien couper. C'est le
    seul endroit où le `O(n²)` se joue -- la géométrie, elle, ne change pas d'un iota.

    Un accélérateur est aussi un ORDRE de stockage : `PowerDiagram_Bsp` range les germes comme
    l'arbre les regroupe (`seed_indices`), et c'est ce rangement, autant que l'élagage, qui fait
    la vitesse (une feuille se lit d'un seul tenant).

    = Le contrat, côté C++

    La CELLULE dirige (`cell/Moteur.h`) : elle demande un demi-espace à un FOURNISSEUR, coupe,
    redemande. Un accélérateur est donc, côté kernel, un fournisseur -- un objet dont la méthode

        template<class Etat> bool suivant( const Etat &e, Local &l, Plane<TK,D> &p );

    remplit le prochain plan et rend `true`, ou `false` quand il n'a plus rien ; `e` est la
    cellule telle qu'elle est DEVENUE (ses sommets, en registres ou en mémoire, voir `cell/Etat.h`)
    et `l` un état que le moteur loge par cellule (la pile d'une descente d'arbre, par exemple).
    C'est là que vivent l'élagage et l'ordre des candidats ; le moteur, lui, n'a aucune politique.

    `AaBsp` est le seul accélérateur aujourd'hui, et son fournisseur est
    `cell/Fournisseurs.h::FournisseurBsp`, qui lit ses tenseurs (`node_box`, `node_begin` /
    `node_end`, `seed_indices`, le majorant affine des poids) et élague par `cell/Elagage.h` --
    exact : une boîte n'est rejetée que si AUCUN sommet de la cellule ne peut être coupé par un
    germe qui s'y trouve. Un autre accélérateur demanderait son propre fournisseur ET son propre
    stockage (`PowerDiagram_Xxx.py` / `.h`, sur le modèle de `PowerDiagram_Bsp`), dont
    `fournisseur<TK>( k0 )` le rend.

    = Le contrat, côté Python

    `nb_seeds()`, pour que le diagramme vérifie que l'accélérateur indexe bien SES germes.
    """

    def nb_seeds( self ):
        """Sur combien de germes il a été construit -- ou `None` s'il ne le sait pas.

        Un accélérateur INDEXE les germes de l'appelant : construit sur un autre nuage, ses indices
        désignent autre chose et la réponse est fausse sans que rien ne le dise. C'est ce que ce
        compte permet de vérifier pour rien du tout, côté appelant.
        """
        return None

