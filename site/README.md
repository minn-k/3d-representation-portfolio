# Portfolio site

Static GitHub Pages source published at https://minn-k.github.io/3d-representation-portfolio/ from the `gh-pages` branch.

- `index.html`: page content and media order
- `style.css`: visual layout
- `assets/`: public images, videos, and PDF files
- `make_web.py`: regenerates the page and media from the local measurement logs when the full research work tree is available

After editing `site/` on `main`, publish its contents with:

```powershell
git subtree split --prefix site -b gh-pages-build
git push origin gh-pages-build:gh-pages --force
git branch -D gh-pages-build
```
