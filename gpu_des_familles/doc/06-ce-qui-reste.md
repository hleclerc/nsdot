# CE QUI RESTE

* **`filnrm8` est la référence 2D** (7.7 ns/germe en uniforme, ×19 ; 19 et 47 sur les lignes,
  `filmix6` à égalité sur Laguerre). Ce qui reste de divergence (6,8 actifs sur 32) est mesuré
  incompressible à peu de frais : l'oracle du tri par coût ([§ profils](05-profils.md)) dit +38 % de lanes au mieux,
  contre une localité par warp qui vaut 2× ; `filreg` / `filregc` à 8 registres et une excursion sont derrière. La piste
  suivante était celle que la seconde passe a révélée : des warps homogènes en taille de
  cellule — mesurée par l'oracle du [§ profils](05-profils.md) : elle ne vaut pas la localité qu'elle coûte. Ce qui reste : la divergence du
  parcours entre les 32 cellules d'un warp (6 actifs) — un tri des cellules par profondeur de
  parcours ou une pile en mémoire partagée ne changeraient pas le fond ; les 3 000 instructions
  par cellule sont à lire ligne à ligne comme pour le 3D.
* **Les paquets, le test en bloc, les deux fils au push, la boucle unique, les lanes
  persistantes, la rotation en mémoire partagée et le tri par coût sont mesurés et perdent**
  ([§ profils](05-profils.md)) : ne pas y revenir sans une idée neuve.
* **`filmsk8` est le même algorithme que `filnrm8` à 96 / 59 registres au lieu de 128 / 74**
  (−25 %), registres triés compris, pour +6 % de temps ([§ profils](05-profils.md)) : c'est le noyau à porter dans le
  noyau des phases, dont le seul frein restant est l'occupation. Le barillet unique y gagne 8 % à
  lui seul ; la cueillette par masques, elle, perd — un prédicat SASS est déjà un masque partagé
  entre `x`, `y` et `c`, et il coûte une instruction au lieu de trois.
* **L'occupation n'est pas le levier en 2D, et raboter les registres fait perdre** ([§ profils](05-profils.md)) :
  `filmsk8` est déjà à 100 % en `float` sans rien forcer et reste derrière `filnrm8` à 75 % ;
  forcer la puissance de deux en dessous fait déborder à tous les coups (jusqu'à +63 %). Le banc
  imprime désormais registres / blocs par SM / occupation / mémoire locale pour chaque variante,
  et `filnrm8c{6,8}`, `filmsk8c{6,8}` refont le balayage. Plancher de bruit du banc : ±2 %, et
  **±5 % sur les lignes Laguerre**.
* **L'ordre dans un registre (`filord8`) et la succession en masques (`filsuc8`) coûtent 21 % de
  registres en moins à vitesse égale** ([§ profils](05-profils.md)) ; leur limite est la longueur de la chaîne de
  dépendances, pas le nombre d'instructions.
  Dans le noyau des phases il manque **trois registres** (131) pour franchir le palier des quatre
  blocs par SM : sortir les `cid` des registres, ou coder `O` autrement, le ferait basculer —
  c'est le chantier le plus court à essayer.
* **À l'échelle ([§ échelle](04-echelle.md)), le noyau tient (+3 % par germe de 10⁶ à 3·10⁷) mais trois choses
  cassent** : la mémoire (33.6 o/germe en `float`, donc 31 Gio à 10⁹ — plusieurs GPU ou un
  découpage spatial), la construction de l'arbre sur CPU (22 s à 3·10⁷, ~15 min à 10⁹, contre 8 s
  de mesure GPU : c'est le `build` qu'il faut porter), et la précision en `float` (erreur par
  cellule en √n, ~1.5 % à 10⁹ — le `double` devient obligatoire). Le schéma par phases perd son
  avance à l'échelle (−14 % à 10⁶, −3 % à 3·10⁷) : sa fenêtre en vol est fixe.
* **Les phases (`filph8`, `filph8g`, `filph8b`, `filph8a`, `filph8c`, `filph8m`) sont écrites et
  mesurées** ([§ profils](05-profils.md)), groupement par feuille compris — par tri, par binning, par écriture directe en
  arène et par arène compactée (la meilleure : −15 % en `double`). Ce qui les bride n'est ni le
  trafic ni l'**occupation** : `filph8m4` atteint 128 registres, quatre blocs par SM et 50 %
  d'occupation sans un octet de débordement, et ne gagne rien. C'est la **latence** des
  allers-retours de l'état en RAM (34–39 cycles de `long scoreboard` par instruction émise) plus
  les barrières entre phases (8–13), avec 0.1 seulement de « pas sélectionné » — il n'y a
  quasiment jamais un warp prêt qui attende son tour : 6,8 → 14,9 lanes actifs, −12 % en `double` sur l'uniforme, mais bornées par la DRAM
  en `float`. **À reprendre en 3D**, où une coupe coûte 800 instructions au lieu de 60 : c'est le
  régime où le trafic est masqué et où le schéma gagne.
* **Revérifier sur une autre carte** : Turing a la plus petite mémoire partagée par SM des
  générations récentes (64 Ko contre 164–228), et un ratio FLOP/octet médian. Deux de nos
  conclusions en dépendent ([§ profils](05-profils.md)) : « la rotation en mémoire partagée perd » et « le trafic des
  phases est masqué ».
* **La coupe 3D à 800 instructions.** Les survivants en place plutôt que renumérotés (moins de
  `rassemble`, mais les trous du CPU à gérer par masques) ; les helpers `rang` / `nieme` appelés
  moins de fois (le nouveau numéro d'un voisin calculé une fois par sommet et non par octet) ;
  moins de petits `if` (chaque `select` sauve trois instructions de convergence).
* **Les facettes**, pour Newton : `cid` est déjà porté par les voies, il manque les aires de faces
  (l'accumulateur par face, en mémoire partagée) et une sortie `( i, j, aire )` par atomique ou par
  compactage.
* **Les poids neufs sans re-téléverser** : `refresh_weights` a son pendant naturel (les majorants
  par nœud, un thread par nœud) ; l'arbre lui-même reste bâti sur l'hôte (`notes/2026-09-02-arbre-morton.md`
  dit pourquoi et par quoi le remplacer sur GPU).
* **`double`.** Sur une carte à FP64 plein (A100 / H100) le rapport `float` / `double` reviendrait
  à celui du CPU ; ici on ne le saura pas.

---

---

[← sommaire](../README.md)
