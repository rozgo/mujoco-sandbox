# Story journals

Each sandbox project has a journal: its story, its videos and its measured results, without an
interactive simulator. The surgical thread robot keeps its interactive journal (`site/`,
`scripts/build_site.py`). The home page (`hub/`) links them all.

```sh
uv run --locked python scripts/build_journals.py            # every journal and the home page
uv run --locked python scripts/build_journals.py dog        # one journal
python3 -m http.server 8791 --bind 127.0.0.1 --directory build/journals
scripts/publish_pages.sh build/journals/dog dog             # publish one journal folder
scripts/publish_pages.sh build/journals/hub .               # publish the home page at the site root
```

## A journal

`site/journals/<id>/journal.json`:

| Field | Content |
| --- | --- |
| `title` | Browser title, two to four words, e.g. "Adaptive Dog Journal" |
| `heading` | The page's h1: what the project is, in one line |
| `eyebrow` | `Engineering journal · <dates> · <main tools>` |
| `lede` | Two or three sentences: the robot, the task, what is learned or programmed |
| `description` | One sentence for search results and the home page |
| `hero` | `{"media": key, "caption": text}`: the main film |
| `metrics` | Four or five `{"value", "label"}` tiles: the measured headline numbers |
| `scope` | One short paragraph of HTML: where the project stands, what is simulated, learned or scripted |
| `footer` | HTML: source link, documentation folder, model licences and attribution |
| `card` | For the home page: `{"kicker", "line", "tags": [...], "poster": repository image path}` |
| `media` | `{key: {"src": repository path, optional "poster" (image path), "poster_t" (s), "start", "duration", "max_width", "crf"}}` |
| `theme` | `{"scheme": "dark" or "light", "theme_color", "tokens": {...}}`, the project's palette below |

`site/journals/<id>/content.html`: chapters only, each
`<section id="slug"><h2>N · Title</h2><p class="when">date · one-line context</p> ... </section>`.
The table of contents is read from these. Media are referenced as `media/<key>.mp4` with
`poster="media/<key>_poster.webp"`, or `media/<key>.webp` for images; the build fails on a reference to
media it did not build.

Components (all styled by `journal.css`):

```html
<div class="prose"><p>…</p></div>
<figure class="film"><video src="media/k.mp4" poster="media/k_poster.webp" controls muted playsinline preload="metadata"></video>
  <figcaption>What it shows, playback speed, what is real.</figcaption></figure>
<figure><img src="media/k.webp" alt="…" loading="lazy"><figcaption>…</figcaption></figure>
<div class="grid2"> two figures </div>   <div class="grid3"> three </div>
<div class="lesson"><strong>Title.</strong> …</div>       <!-- a lesson learned -->
<div class="failure"><strong>Title.</strong> …</div>      <!-- something that failed, kept on record -->
<div class="note-box"><strong>Title.</strong> …</div>     <!-- a check, a caveat -->
<div class="result-strip"><div class="tile"><div class="big">60<small>/ 60</small></div>
  <div class="what">…</div><div class="vs">…</div></div> … up to four tiles</div>
<div class="spec-cards"><div class="card"><h4>…</h4><ul><li>…</li></ul></div> …</div>
<div class="table-wrap"><table><tr><th>…</th><th class="num">…</th></tr><tr class="hl">…</tr></table></div>
<ol class="next-steps"><li><strong>Title</strong>Text.</li> …</ol>
```

## Rules

- Every number comes from a file in the repository (reports, JSON, docs); write what was measured,
  with its conditions. Keep failures on record. Distinguish learned, scripted and simulated parts, and
  rendered video from generated video. State video playback speed.
- Plain, concise, public-safe: no machine connection details, credentials, private names or links to
  private material.
- Videos are re-encoded for the web; keep a journal under about 70 MB built.

## Palettes

Each journal has its own palette; none reuses the surgical journal's rose and lavender.

| Journal | Scheme | Character | Key colours |
| --- | --- | --- | --- |
| Adaptive dog | dark | graphite, teal, orange | `#08090B` `#131518` `#EDEFF1` `#49D6C5` `#FF6B23` |
| Neural wind | dark | storm navy, cyan, parcel orange | `#10202B` `#1D4A63` `#65E2D2` `#91B3E7` `#FF8C28` |
| RC rovers | dark | night, signal green, amber | `#101B24` `#133A33` `#3BE2AC` `#F9BF2D` `#D96B63` |
| Hexapod transfer | light | workshop paper, walnut, mug orange, block blue | `#F6F1E8` `#6B4A2F` `#E06D14` `#3B7BEF` `#1E1A16` |
| Amphibious | light | sea teal, foam, sand, float orange | `#0E2A33` `#1F7A94` `#F3F7F6` `#C9B28A` `#E8901F` |

The full token sets are in each `journal.json`.
