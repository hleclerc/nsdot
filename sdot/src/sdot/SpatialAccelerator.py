from loom.util import Aggregate


class SpatialAccelerator( Aggregate ):
    """Ce qui répond à « QUELS germes valent la peine d'être essayés, et dans quel ordre ».

    Un accélérateur ne sait pas ce qu'est une cellule. Il connaît la répartition des germes dans
    l'espace, et il en tire une ÉNUMÉRATION : au lieu des `n - 1` bissectrices que
    `PowerDiagram.cxx::make_cell` essaie une par une, il propose les germes proches d'abord et
    s'arrête d'explorer une région dès que l'appelant lui dit qu'elle ne peut plus rien couper.
    C'est le seul endroit où le `O(n²)` se joue -- la géométrie, elle, ne change pas d'un iota.

    = Le contrat, côté C++

    La structure C++ engendrée par la sous-classe doit offrir :

        void for_each_candidate( const auto &from, SI i0, auto &&scratch,
                                 auto &&may_cut, auto &&cut_with ) const;

    `from` est le point d'où l'exploration rayonne (le germe `i0`), `scratch` le tampon de
    travail que `thread_scratch` a déclaré, et les deux derniers arguments sont des CALLBACKS
    fournis par l'appelant :

    - `may_cut( lo, hi, wa, wb ) -> bool` : « un germe posé n'importe où dans la boîte
      `[ lo, hi ]`, de poids majoré par `wa . y + wb`, pourrait-il encore entamer ce qu'il reste
      de la cellule ? ». CONSERVATIF : il ne rend `false` que lorsqu'il est sûr. C'est l'appelant
      qui tient la cellule, donc c'est lui qui répond ;
    - `cut_with( i1 ) -> bool` : applique la coupe du germe `i1`. `false` = tout arrêter (capacité
      dépassée, ou cellule devenue vide).

    L'accélérateur doit visiter TOUT germe que `may_cut` n'a pas exclu, `i0` excepté. Rien de
    plus : ni l'ordre exact, ni le fait de repasser deux fois sur un germe (ce serait seulement
    du travail perdu) ne font partie du contrat. Ce qui en fait partie, et qui est la seule chose
    dont dépend la CORRECTION, c'est de ne jamais taire un germe que `may_cut` a admis.

    La forme de la région est donc, aujourd'hui, une boîte alignée sur les axes plus un majorant
    AFFINE des poids -- ce que `AaBsp` produit. Un accélérateur qui bornerait ses régions
    autrement (sphères, plans obliques) demanderait d'élargir ce couple, pas de le remplacer :
    `may_cut` est écrit une fois, chez l'appelant, et c'est lui qui saurait quoi faire de la
    nouvelle forme.

    = Le contrat, côté Python

    Deux méthodes, toutes deux avec un défaut vide, parce qu'un accélérateur n'a pas forcément
    besoin de mémoire de travail :

    - `thread_scratch( num_thread )` -- le tampon PAR WORK-ITEM dont la marche a besoin, ou
      `None` ;
    - `bytes_per_thread()` -- ce qu'il pèse, pour que l'appelant en tienne compte quand il décide
      combien de work-items il peut se permettre.
    """

    def nb_seeds( self ):
        """Sur combien de germes il a été construit -- ou `None` s'il ne le sait pas.

        Un accélérateur INDEXE les germes de l'appelant : construit sur un autre nuage, ses indices
        désignent autre chose et la réponse est fausse sans que rien ne le dise. C'est ce que ce
        compte permet de vérifier pour rien du tout, côté appelant.
        """
        return None

    def thread_scratch( self, num_thread ):
        """Le tampon de travail d'UN work-item, batché sur `num_thread` -- ou `None`."""
        return None

    def bytes_per_thread( self ):
        """Ce que `thread_scratch` immobilise par work-item, en octets."""
        return 0
