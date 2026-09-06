# ADELE — project page

Source for the ADELE project page, served with GitHub Pages at
<https://johannes-weidenfeller.github.io/adele>.

Single static file: `index.html` (styles inlined, no build step, no JS).

## To finish the page

1. Drop figures into `static/images/`:
   - `teaser.jpg` — headline result
   - `method.jpg` — pipeline diagram
   - `results.jpg` — qualitative comparisons
   Then replace each `<div class="placeholder">…</div>` with
   `<img src="static/images/teaser.jpg" alt="…">`.
2. Fill in the two `href="#"` link buttons (paper PDF, arXiv) and the arXiv id in
   the BibTeX block.
3. Swap the "Code coming soon" span for a link to the code repo once it is public.

## Publishing

    git init && git add -A && git commit -m "ADELE project page"
    git branch -M main
    git remote add origin git@github.com:johannes-weidenfeller/adele.git
    git push -u origin main

Then: repo Settings → Pages → Source = `main`, folder = `/ (root)`.
