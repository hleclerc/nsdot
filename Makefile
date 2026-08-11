# Point d'entrée dev du monorepo nsdot — orchestre les 3 projets indépendants.
# Chaque sous-projet a son propre pyproject.toml et s'installe séparément.
# Ordre d'installation : loom → sdot → otrec (dépendances).
#
# Les cibles personnelles/distantes (lmo, cuda, apptainer) vivent dans .private/Makefile,
# qui `include` ce fichier.
#
# Prérequis : une fois, `make install` (editable installs dans l'env). Ensuite `import loom`,
# `import sdot`, `import otrec` marchent sans PYTHONPATH, ici comme depuis n'importe quel cwd.

# On suppose l'environnement DÉJÀ ACTIVÉ (venv / conda / micromamba / …) : on appelle `python`
# nu. `RUN` est un hook d'override optionnel pour préfixer la commande sans activer l'env :
#   make test RUN='micromamba -n vfs run'
# (le .private/Makefile s'en sert pour le wrapping personnel micromamba.)
RUN ?=
PY  ?= python

# Réglages consommés par la couche de compilation JIT (loom/compilation/adaptive_cpp).
export SDOT_XMAKE_MODE  ?= debug
export SDOT_FORCE_BUILD ?= 0
export SDOT_FTYPE       ?= FP64

# Filtre de test (pytest -k) :
#   make test T=Cell
#   make test T='test_basic'
T ?=

.DEFAULT_GOAL := test
.PHONY: install test test-loom test-sdot test-otrec run clean help

install: ## Install editable des 3 packages dans l'ordre
	$(RUN) $(PY) -m pip install -e ./loom[jax,torch]
	$(RUN) $(PY) -m pip install -e ./sdot
	$(RUN) $(PY) -m pip install -e ./otrec

test: test-loom test-sdot test-otrec ## Tous les tests

test-loom: ## Tests loom (core)
	$(RUN) $(PY) -m pytest loom/tests -k "$(T)" $(PYTEST_ARGS)

test-sdot: ## Tests sdot (transport optimal)
	$(RUN) $(PY) -m pytest sdot/tests -k "$(T)" $(PYTEST_ARGS)

test-otrec: ## Tests otrec (reconstruction CT)
	$(RUN) $(PY) -m pytest otrec/tests -k "$(T)" $(PYTEST_ARGS)

run: ## Lance l'app de reconstruction (plot)
	$(RUN) $(PY) -m otrec.viz.plot

clean: ## Supprime les caches Python et le build
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf build/

help: ## Liste les cibles
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
