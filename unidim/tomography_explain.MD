2. Une analogie très simple : les ombres

Imagine un objet devant une lampe.

Tu regardes son ombre depuis une direction :

Objet                 Ombre

  ███                    ███
 █████       →          █████
  ███                    ███

Tu obtiens une information sur l'objet, mais seulement dans une direction.
Tu obtiens plusieurs ombres/projections.

Avec suffisamment de directions, tu peux essayer de retrouver la forme de l'objet.

C'est essentiellement l'idée physique derrière la tomographie.

# Reconstruction ici

Ton entrée principale est le sinogramme.
```python
sino = Sinogram(
    CtGeometry(
        nb_angles=600,
        nb_bins=4096,
        extent=2.0
    )
)
``` 

Le sinogramme contient des mesures pour :

600 angles différents
4096 positions/détecteurs pour chaque angle.

On peut le représenter :

                 positions du détecteur
             ───────────────────────────→

angle 0°      █ █ █ █ █ █ █ █ █ █ █ █ █
angle 1°      █ █ █ █ █ █ █ █ █ █ █ █ █
angle 2°      █ █ █ █ █ █ █ █ █ █ █ █ █
...
angle 599°    █ █ █ █ █ █ █ █ █ █ █ █ █
       │
       ↓
     angles

Donc ton entrée est essentiellement une matrice :

$$ 600 \times 4096 $$

Chaque valeur correspond à une mesure.

4. Mais qu'est-ce qu'une ligne du sinogramme ?

Prenons un seul angle.

Par exemple, tu regardes l'objet horizontalement :

       détecteurs
   ↓   ↓   ↓   ↓   ↓

        █████
      ██     ██
      ██     ██
        █████

Chaque détecteur reçoit une information correspondant à ce qui se trouve le long d'une certaine ligne traversant l'objet.

On obtient quelque chose comme :

détecteur :

1   2   3   4   5   6   7   8
│   │   │   │   │   │   │   │
0   0   2   5   5   2   0   0

C'est une projection 1D de l'objet 2D.

7. Mais ton code fait quelque chose d'un peu particulier

Ton programme ne reconstruit pas directement une image composée de pixels.

Il représente l'objet avec des points.

C'est fondamental pour comprendre ton code.

Au lieu de dire :

pixel pixel pixel pixel
pixel pixel pixel pixel
pixel pixel pixel pixel

il dit :

     •
           •
  •
                •
       •
 •

Chaque point est un Dirac.

Donc si tu as :

nb_points_final = 100_000

la sortie est essentiellement :

100 000 positions (x, y)

8. Pourquoi des points ?

Imagine que tu veuilles représenter un disque.

Avec des pixels :

████████
████████
████████
████████

Avec des Dirac :

 • • • •
• • • • •
 • • • •
• • • • •

Chaque point représente une petite quantité de masse/densité.

Avec suffisamment de points, leur distribution reproduit l'objet.

Dans ton code :

points.shape

est donc typiquement :

(100000, 2)

Les deux colonnes sont :

x   y
9. Comment sait-il où mettre les points ?

C'est là que la reconstruction devient intéressante.

Au départ, le programme place les points aléatoirement :

     •       •
  •       •
        •
 •            •
       •

Puis il se demande :

"Si mes points étaient réellement l'objet, quelles projections produiraient-ils ?"

10. Projection d'un point

Prenons un point :

          • (x,y)

Pour un angle donné, on calcule sa projection sur une ligne :

             •
            /
           /
          /
---------•------------→

Mathématiquement, c'est :

points @ normal

Cela transforme les coordonnées 2D :

(x, y)

en une coordonnée 1D :

projection
11. On compare avec le sinogramme réel

On a maintenant deux choses.

Ce que l'on prédit avec nos points :
projection des 100 000 points
Ce que l'on mesure réellement :
ligne correspondante du sinogramme

Le programme compare les deux.

         notre reconstruction
                  │
                  ↓
             projection
                  │
                  │
                  ▼
             ┌─────────┐
             │ COMPARER│
             └─────────┘
                  ↑
                  │
                  │
             mesure réelle
             du sinogramme

La différence est la loss.

12. Dans ton code, la comparaison est Wasserstein

C'est ici :

_w2_1d(...)

Le programme utilise une distance de Wasserstein.

Très intuitivement, Wasserstein répond à une question du genre :

Combien faudrait-il déplacer la masse de ma projection pour qu'elle ressemble à la mesure réelle ?

Par exemple :

Notre projection :

        ████
      ███████
         ██


Mesure réelle :

          ████
        ███████
           ██

Les deux distributions se ressemblent, mais sont légèrement décalées.

Wasserstein mesure en quelque sorte le coût du déplacement de la masse.

13. Et ensuite ?

Le programme calcule :

         erreur
           ↓
       gradient
           ↓
   "dans quelle direction
    déplacer les points ?"
           ↓
      déplacement
           ↓
       nouveaux points

Par exemple :

Avant :

 •        •
      •
   •        •

        ↓ optimisation

Après :

    • • •
  • • • • •
    • • •

Les points se déplacent progressivement pour produire des projections qui ressemblent davantage au sinogramme.


16. Dans ton exemple concret

Tu construis artificiellement un objet avec :

sino.add_disk(
    center=[0, 0],
    radius=0.9,
    density=+1.0
)

puis :

sino.add_disk(
    center=[0, 0],
    radius=0.7,
    density=-1.0
)

Donc tu fabriques un objet du genre :

       +++++++++
     +++++++++++++
    +++++-----+++++
   +++++-------+++++
   +++++-------+++++
    +++++-----+++++
     +++++++++++++
       +++++++++

C'est essentiellement :

$$ \text{grand disque} - \text{petit disque} $$

Donc une structure annulaire.

Le programme connaît le sinogramme produit par cet objet, mais lorsqu'il lance :

multiscale_optimize(...)

il cherche à retrouver l'objet à partir de ces projections.

17. Entrées et sortie, très simplement
Entrées

Tu as essentiellement :

1. Le sinogramme

600 angles × 4096 mesures

2. La géométrie

angles
position des détecteurs
taille du domaine

3. Le nombre de points souhaité

100 000
Calcul interne

Le programme fait :

points 2D
   ↓
projection selon chaque angle
   ↓
distribution 1D
   ↓
comparaison au sinogramme
   ↓
Wasserstein
   ↓
loss
   ↓
gradient
   ↓
L-BFGS
   ↓
nouveaux points
Sortie
points = multiscale_optimize(...)

donne :

100 000 × 2

donc :

[
 [x1, y1],
 [x2, y2],
 [x3, y3],
 ...
 [x100000, y100000]
]

Ces points constituent l'objet reconstruit.

18. Le point essentiel à retenir

Tu peux résumer tout ton programme par cette phrase :

Je connais les projections 1D d'un objet 2D sous 600 angles différents, et je cherche 100 000 points 2D dont les projections reproduisent au mieux ces mesures.

C'est ça, la reconstruction.

Et le rôle de chaque morceau de ton code devient alors beaucoup plus clair :

Élément	Rôle physique
sino	Les mesures/projections du CT
normals	Les différentes directions d'observation
bin_mass	La distribution mesurée sur chaque projection
points	L'objet que l'on cherche à reconstruire
points @ normal	Projection de l'objet dans une direction
_w2_1d	Compare projection calculée et projection mesurée
loss	Erreur totale sur les 600 angles
gradient	Indique comment déplacer les points
L-BFGS	Déplace intelligemment les points
jax.jit	Accélère tous ces calculs sur GPU
multiscale	Commence avec peu de points puis augmente progressivement
tracker	Enregistre l'évolution de la reconstruction

Et il y a une distinction très importante : ton code ne fait pas la reconstruction CT classique par FBP (Filtered Back Projection). Il utilise une reconstruction itérative basée sur une représentation par Dirac et une fonction coût Wasserstein. C'est justement ce qui rend ce code intéressant.