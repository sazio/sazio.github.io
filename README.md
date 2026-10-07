# sazio.github.io

Personal website of Simone Azeglio: <https://sazio.github.io>.

Served by GitHub Pages from `master`. Every push rebuilds the site in about a minute.

## Layout of the repo

| Path | What it is |
| --- | --- |
| `index.html` | The homepage. Static HTML with a short Jekyll front matter (permalink `/`, redirect from `/about/`). |
| `assets/home/data.js` | **All homepage content**: research cards, publications, workshops, news, posts list, links. |
| `assets/home/style.css` | Homepage styles and the colour tokens (dark by default, light via the toggle). |
| `assets/home/mosaic.js` | The live retinal mosaic in the header (DoG receptive fields, ON/OFF cells, spike raster). |
| `assets/home/main.js` | Renders `data.js` into the page. |
| `assets/home/theme.js` | Light/dark toggle, shared with the post pages. |
| `_posts/` | Blog posts (Markdown). Images used by older posts live in `_posts/img/`. |
| `_layouts/post.html`, `assets/home/post.css` | Post layout, matching the homepage. |
| `_pages/` | Workshop pages (`/workshops/...`), `/projects/`, archives. These still use the original theme. |
| `_publications/`, `_talks/` | Older per-item pages (`/publications/...`, `/talks/...`). Not linked from the homepage. |
| `files/` | PDFs. `files/cv.pdf` is the CV linked in the header. |
| `images/` | Images, including `profile.jpg` (homepage portrait). |

## Common edits

**Add a paper, news item or workshop.** Edit the matching array in `assets/home/data.js` and push. Each entry is a small object; copy a neighbour. Publications with `selected: true` show by default, the rest under "All". `highlight` adds the coloured badge (e.g. `"Oral"`).

**Update the CV.** Replace `files/cv.pdf`.

**Write a post.** Add `_posts/YYYY-MM-DD-title.md` with front matter:

```yaml
---
title: 'Post title'
date: 2026-10-07
permalink: /posts/2026/10/post-title/
tags:
  - tag one
---
```

The post layout is applied automatically. Add it to the `writing` list in `data.js` so it appears on the homepage.

## Previewing locally

The homepage is plain HTML, so the quickest preview is a static server from the repo root:

```bash
python3 -m http.server 8000   # then open http://localhost:8000/index.html
```

The front matter at the top of `index.html` shows up as text in this preview; that is expected. Posts and the other pages need Jekyll (`bundle install && bundle exec jekyll serve`, which requires Ruby 3.x).

## Credits

The pages that still use the old design are built on [Academic Pages](https://github.com/academicpages/academicpages.github.io), itself a fork of the [Minimal Mistakes](https://mmistakes.github.io/minimal-mistakes/) Jekyll theme © Michael Rose, released under the MIT License (see `LICENSE`).
