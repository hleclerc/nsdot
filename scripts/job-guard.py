#!/usr/bin/env python3
"""Cale de transition : les sessions ouvertes avant le 02/10/2026 ont encore `~/.claude/hooks/job-guard.py`
( un lien sur ce fichier ) en memoire. Le garde-fou est `errand-guard.py` ; ce fichier lui passe la main.
A supprimer, avec le lien, quand plus aucune session ancienne ne tourne."""
import os, runpy
runpy.run_path( os.path.join( os.path.dirname( os.path.realpath( __file__ ) ), "errand-guard.py" ), run_name = "__main__" )
