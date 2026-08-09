#!/usr/bin/env python3
"""Add dark mode JS wiring to points_html.py."""
path = '/Users/hugo.leclerc/nsdot/applications/reconstruction/viz/points_html.py'
with open(path, 'r') as f:
    content = f.read()

# Edit 1: make ctx.fillStyle theme-aware in draw() function
content = content.replace(
     "ctx.fillStyle = '#000000';",
      "var dotColor = DARK ? '#ffffff' : '#000000';\n" +
       "ctx.fillStyle = dotColor;"
)

# Edit 2: Add theme detection before resize() call  
content = content.replace(
     "// dernière frame par\n    default (l'état final/convergé, ce qu'on veut voir en premier)\n",
      None  # will be replaced below
)
