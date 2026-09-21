# 2026-09-22 — `OtPlan` : le solveur du banc, en C++, en un appel

Ce que `solvers_des_familles` a conclu (README § 3, § 7, § 9, § 10, § 11) est maintenant CE QUE
`sdot.OtPlan` fait, et rien d'autre : tout l'ajustement — départ, Newton amorti, diagrammes,
laplacien, solveur linéaire, pas par les limites, continuation en largeur — tourne dans **un seul
`driver.call`** (`sdot/include/sdot/otplan/`). Python pose le problème (un `PowerDiagram`, les masses
cibles, les options) et lit ce qui sort (les poids, des statistiques, un historique) : plus de
numpy dans `OtPlan.py`, plus de L-BFGS, plus de moindre-carrés, plus de « dual par descente ».

## Ce qui a été retenu du banc, et où ça vit

| conclusion du banc | ici |
|---|---|
| Newton amorti (KMT) gagne partout, 2–4× contre L-BFGS / CG (§ 10) | `otplan/Newton.h` — le seul solveur |
| le laplacien assemblé SANS TRI depuis les facettes, `c_ij = ∫_facette ρ / 2|p_i − p_j|` | `otplan/Laplacien.h`, les facettes lues dans le même balayage que les masses (`otplan/Balayage.h`) |
| Cholesky (Eigen, AMD, motif gardé) en 2D, AMG (AMGCL, Ruge-Stüben+GS) au-delà (§ 3) | `otplan/Lineaire.{h,cpp}` — une UNITÉ DE DOMAINE (`FfiCode( sources = [...] )`), `AUTO` = Cholesky en 2D ≤ 3e5, AMG sinon, CG maison en repli |
| le pas repart du dernier pas accepté, jamais de 1 | `NewtonOptions::mult_ok = 4` |
| `essai-limites` en 2D : l'essai `β`, puis les limites des seules cellules sous `ε`, `t = 0.9 α*` ; `β₀ = 1/4`, ×2 après un essai direct (§ 7, § 9.7) | `otplan/Limites.h` — le polynôme (Lebesgue) ou la bissection en masse (toute densité), cellules exactes à chaud |
| le fournisseur « à alpha » : `w + α d` sans rafraîchir l'arbre, majorant `a_w + α a_d`, départ à chaud sur les voisins | `PdAlphaBsp` : le stockage vu décalé, parcouru par le `FournisseurBsp` ORDINAIRE, la liste chaude comme mémoire (`MEMO`) |
| la continuation en largeur de convolution, ratio `√2`, depuis `0.5 × diamètre`, jusqu'à `σ_min / 4` puis 0 (§ 9) | `otplan/Continuation.h` — gaussiennes : `σ' = √(σ² + s²)` ; image : flou séparable sur sa grille |
| la tangente `dw/ds`, l'ordre 2, la garde par cellule, la barrière : n'apportent rien avec les limites (§ 9.4–9.7) | pas repris |
| le multi-échelle, le glissement, l'homotopie, MAN, le pas tensoriel (§ 7.3–7.5, § 8) | pas repris |
| `double` pour le noyau (§ 4) | `kernel_dtype = "FP64"` par défaut dans `OtPlan` |

## L'architecture

```
sdot/include/sdot/otplan/
  Laplacien.h     CSR des hors-diagonaux + diagonale, `crs_reduit` (jauge w_0 = 0)
  Lineaire.h/.cpp SolveurLineaire : Cholesky / Amg / Cg -- compilé une fois, lié par le noyau
  Balayage.h      set_weights( w ) ( poids triés + majorants ), mesures( a, &facettes ) ; scratch géré ici
  Newton.h        la boucle, ESSAIS ou ESSAI_LIMITES, l'historique par `apres_pas`
  Limites.h       PdAlpha*, FournisseurListe, PolyCellule, Limites2D::alpha_min
  Continuation.h  Convolee<Dist>::at( s ), etapes( s0, ratio, s_min )
  Solve.h         resoudre<TK>( queue, pd, dom, dist, nu, w0, options, weights, hist, stats )
sdot/src/sdot/OtPlan.py   la classe ; hull.py : l'enveloppe par demi-espaces d'appui
```

**Les poids sont des SORTIES de l'appel.** Les entrées d'un `driver.call` sont en lecture seule ;
le solveur pose des poids à chaque essai. `PowerDiagram_Bsp.h::with_weights( sorted_weights, node_wa,
node_wb )` rebâtit le stockage sur trois vues de sortie (l'ordre des membres est celui de
`kernel_form`), et `PowerDiagram_Bsp._solver_weights_after` les adopte après l'appel.

**Le scratch des cellules est géré par le solveur**, pas par loom : un appel enchaîne cent
diagrammes, et un débordement ne doit relancer que le balayage en cours (`Balayage::mesures`
double sa ligne par fil et recommence). Les passes de limites ont leur propre scratch par fil.

**Le départ** (`Solve.h`) : les poids donnés ; s'ils vident une cellule (< 1e-3 min ν), le
Voronoï s'il fait mieux ; si le Voronoï laisse une cellule vide (des germes hors du domaine), la
similitude qui ramène le nuage dans le pavé. `stats["depart"]` le dit.

**La masse cible** est remise à l'échelle de ce que le domaine contient (`Σ a` ne dépend pas des
poids) — à chaque étape de la continuation. Le plan est celui vers la densité restreinte au
domaine, normalisée.

**La mémoire 3D** (`memory = K`, 32 par défaut en 3D) est refaite DANS le solveur à chaque
balayage (`Balayage::mesures` → `diagram::memorise`), dans deux vues de sortie initialisées depuis
les souvenirs d'avant ; un souvenir d'un essai refusé reste exact (il ne fait qu'ordonner les coupes).

**Le domaine non borné** (des gaussiennes sans `boundaries`) est fermé par l'ENVELOPPE des diracs,
approchée par l'extérieur : 16 demi-plans d'appui en 2D (les axes compris → un pavé de départ +
12 coupes par cellule), 26 en 3D (`sdot/hull.py`). L'enveloppe exacte peut avoir `n` arêtes et
chaque cellule est coupée par tous les plans du domaine : elle rendrait le diagramme quadratique.

**La continuation** se déclenche en `auto` quand le meilleur départ laisse une cellule sous
`1e-2 × min ν` (`conv_threshold`) — les cas du banc où Newton direct stagne ont des cellules à
`1e-17`.

## Les chiffres (2026-09-22, machine partagée, 8 fils, `job -q`)

| cas | it / diag (reculs) | temps C++ | banc (README) |
|---|---|---|---|
| 2D uniforme 1e5 | 6 / 9 (2) | 2.4 s (diag 0.47, Cholesky 1.85) | 8 diag (6 it), 2.2 s |
| 2D lignes 1e5, s = 0.005, essais KMT (`step="trials"`) | 23 / 56 (32) | 8.8 s | KMT 24 / 113, 17.9 s |
| 2D lignes 1e5, `step="limits"` (défaut) | **19 / 32 (0)** | 6.6 s (diag 2.9, lin 3.3) | essai-limites 19 / 40, 8.4 s |
| 3D uniforme 1e5 | 4 / 7 (2) | 8 s (diag 3.5, AMG 4.5 sans OpenMP) | 6 / 9, 5.2 s |
| 3D uniforme 2e4, `memory = 0` → 32 (le défaut) | 5 / 7 | 0.107 → 0.090 s par diagramme | § 11 : −25 à −40 % à 1e5+ |
| densité 4 gaussiennes σ = 0.05, n = 5000, direct | STAGNATION | | idem |
| … continuation auto | 14 étapes, 57 / 77 (1) | 1.8 s | |
| densité σ = 0.02, n = 1e5, continuation | 17 étapes, 156 / 246 (2) | 113 s (diag 69, lin 40, lim 1.7) | 15 étapes, 130 / 204, 108 s |

Le diagramme d'une densité gaussienne coûte 0.28 s à 1e5 (la réduction exacte par triangles de
`SumOfGaussians`) contre 0.09–0.13 dans le banc (la circulation) : c'est le poste à reprendre si
les densités deviennent le cas courant.

Le temps Python autour de l'appel : ~5 s à 1e5 pour une NOUVELLE forme (le traçage Jax de chaque
`driver.call` — arbre par niveau, majorants, domaine —, 0.24 s par appel), 0.3 s à forme égale.
C'est le sur-coût par appel de loom, pas le solveur.

## Ce qui reste

- **CUDA** : le solveur est du code hôte sur `CpuQueue` (les balayages, le laplacien, Eigen/AMGCL).
  Sur un driver GPU, `OtPlan` lève `NotImplementedError` ; le chemin naturel est de laisser les
  balayages sur le device et le reste sur l'hôte — une queue par étape, pas un `if`.
- **Les limites en 3D** (le banc ne les a pas écrites) : `step = "trials"` y reste le défaut.
- **Les gaussiennes en 3D** : `facet_mass` n'est écrit qu'en 2D (le laplacien d'une densité
  ponctuelle au-delà demande une quadrature de face).
- **AMGCL sans OpenMP** : l'unité de domaine est compilée avec les flags de loom ; `-fopenmp`
  rendrait la hiérarchie parallèle (4.5 s → ~2 s en 3D à 1e5).
- **Le relèvement minimal** (§ 8.4–8.5, l'enveloppe convexe de `(1−ε)p² − w`) : un départ
  admissible meilleur que la similitude quand les poids donnés vident des cellules ; le Voronoï
  suffit aujourd'hui.
- `scripts/job` : un travail tué (`kill`) laisse son `flock` orphelin, qui garde `queue.lock` —
  plus aucun travail générique ne passe. `flock -x 8 9>&-` (fermer le descripteur 9 dans l'enfant)
  suffirait.
