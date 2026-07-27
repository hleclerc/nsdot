# Point d'entrée dev de nsdot -- générique et sans dépendance machine.
# Les cibles personnelles/distantes (lmo, cuda, apptainer) vivent dans .private/Makefile,
# qui `include` ce fichier.
#
# Prérequis : une fois, `make install` (editable install dans l'env). Ensuite `import sdot`
# marche sans PYTHONPATH, ici comme depuis n'importe quel cwd.

# On suppose l'environnement DÉJÀ ACTIVÉ (venv / conda / micromamba / …) : on appelle `python`
# nu. `RUN` est un hook d'override optionnel pour préfixer la commande sans activer l'env :
#   make test RUN='micromamba -n vfs run'
# (le .private/Makefile s'en sert pour le wrapping personnel micromamba.)
RUN ?=
PY  ?= python

# Réglages consommés par la couche de compilation JIT (compilation/adaptive_cpp).
export SDOT_XMAKE_MODE  ?= debug
export SDOT_FORCE_BUILD ?= 0
export SDOT_FTYPE       ?= FP64

# Filtre de test, aligné sur le harnais (nom, "Fichier::nom", ou "[tag]") :
#   make test T=Cell
#   make test T='[fast]'
T ?=

.DEFAULT_GOAL := test
.PHONY: install test run clean help

install: ## Editable install de sdot dans l'env
	$(RUN) $(PY) -m pip install -e .

test: ## Tests C++ + Python (filtre optionnel T=...)
	$(RUN) $(PY) scripts/run_tests.py $(T)

run: ## Lance l'app de reconstruction (plot)
	$(RUN) $(PY) -m applications.reconstruction.viz.plot

clean: ## Supprime les caches Python
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

help: ## Liste les cibles
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'
