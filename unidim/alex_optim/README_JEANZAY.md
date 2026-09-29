# JAX sur Jean-Zay H100

Procédure pour utiliser JAX sur H100. Adapter `/chemin/vers` à ses chemins.

## 1. Créer le venv

À faire sur terminal sur la frontale, car les noeuds de calcul n'ont pas accès à Internet.

```bash
module load arch/h100
module load pytorch-gpu/py3/2.8.0

python -m venv /chemin/vers/env_nsdot_jax_pytorch2.8
source /chemin/vers/env_nsdot_jax_pytorch2.8/bin/activate

python -m pip install --upgrade pip
python -m pip install -r /chemin/vers/nsdot/unidim/alex_optim/requirements.txt
deactivate
```

## 2. Activer l'environnement

Dans cet ordre :

```bash
module load arch/h100
module load pytorch-gpu/py3/2.8.0
source /chemin/vers/env_nsdot_jax_pytorch2.8/bin/activate
```
