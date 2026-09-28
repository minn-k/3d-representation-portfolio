# Portfolio site

Static page published at https://minn-k.github.io/3d-representation-portfolio/ (branch `gh-pages`, the contents of this folder).

`make_web.py` rebuilds `index.html` and `assets/` from the local work tree (`APG_ROOT`): every number on the page is read
from the measurement logs in `genai/out/*/*.json` (mirrored in `demos/generative-3d/results`). Videos are re-encoded to
H.264 with faststart.

Publish: `git subtree split --prefix site -b gh-pages-build && git push origin gh-pages-build:gh-pages --force`.
