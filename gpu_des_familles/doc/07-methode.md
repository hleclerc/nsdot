# COMMENT LES CHIFFRES SONT PRIS

Avant chaque chrono, 300 ms de noyau en boucle : le CPU (l'arbre, le témoin) laisse le GPU
redescendre en fréquence, et un seul tour de chauffe ne le remonte pas — le même noyau donnait
17 puis 13 ns/germe selon qu'il passait premier ou second. Puis 10 tours, minimum.

`job -b -- ./build/linux/x86_64/release/mesures --threads 8 --reps 3 --reps-gpu 10 --kernel float`
puis la même en `double` — machine seule, rien d'autre ne tourne. Un `PowerDiagram` CPU bâtit
l'arbre ; le témoin mesure (chauffe, minimum de 3) ; `DiagrammeGpu` reçoit l'arbre ; chaque
variante fait un tour de chauffe puis 10 tours chronométrés par événements CUDA autour du (ou des
deux) noyau(x), et le minimum est gardé. Les profils sont pris avec `ncu` (CUDA 13.3) sur la
première passe (`--kernel-name-base demangled --kernel-name "regex:voies<\(bool\)0, \(int\)2, float>"`)
et la répartition par ligne source vient de `--page source --print-source sass,cuda --csv`,
agrégée par un petit script.

---

[← sommaire](../README.md)
