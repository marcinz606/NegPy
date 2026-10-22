---
name: edge-print-from-photo
description: "Use when a user supplies a photo or scan of film negatives or slides that shows the rebate (the film edge with perforations, frame numbers, maker name, DX barcode or KEYKODE) and wants NegPy's contact sheet edge print for that stock to match, offers a stock's edge codes or DX number, or reports that a stock's edge print looks wrong. Keywords: edge print, rebate, edge markings, frame numbers, DX code, KEYKODE, respool, film_stocks.json, contact sheet, crowdsource."
---

# Edge print from a photo

NegPy draws a film's edge print from two places:

| What | Where | Changed by |
|---|---|---|
| A maker's layout: which marks repeat where, sizes, DX or not | `LAYOUTS` in `negpy/services/film_edge/layouts.py` | a layout entry, shared by every stock that names it |
| One stock's words, codes and DX number | the `edge` object of its entry in `gear/film_stocks.json` | data only |

Most photos need data only. A layout changes when the photo shows a structure no existing layout has.

## 1. Find the stock and what NegPy prints now

    grep -n '"stockName"' gear/film_stocks.json | grep -i <name>
    uv run python .claude/skills/edge-print-from-photo/render_edge.py <stock-id> --out /tmp/before.png

The script prints the resolved style and every item in roll millimeters (frame 1 centered at 19.0 mm on 135).
`--edge '{"text": "...", "style": "..."}'` tries values without editing the catalog. A respooled film
(CineStill, Silbersalz35, Candido, any "Vision3 ... with the remjet removed") prints its source stock: give
it `{"printsAs": "<source id>", "source": "..."}` and stop.

## 2. Read the film

Read the photo at full size; crop and enlarge the rebates if text is small. Record per edge:

- **Text, exactly as printed**: words, spaces, hyphens, generation numbers ("GB 200-7", "HP5 PLUS"). Not the box,
  not the datasheet, not the catalog name. The renamed Kodak stocks keep their legacy marks.
- **Repeats**: once a frame, every Nth frame (batch codes), a fixed pitch unrelated to the frames (120 numbers,
  cine keys). For marks that alternate, note which one follows the odd frame numbers and which the even.
- **Marks**: triangle, outline triangle, arrow under or before the A label, dots, zero-frame dot.
- **Barcodes**: DX (a short two-row code after each frame number), KEYKODE (letters + 10 digits + dot, then a
  long single-row code), or none. "None" is a finding: it needs a layout without DX. Decode what is there:

      uv run --no-project --with zxing-cpp --with pillow python -I .claude/skills/edge-print-from-photo/decode_barcodes.py PHOTO

  It prints each DX code as part1-part2/frame. The frame numbers must agree with the printed labels.

## 3. Measure, if a position looks wrong

- Scale: the perforation pitch is 4.75 mm (135). Count holes over 8 or more pitches; use the nearest holes for
  each measurement, which cancels most perspective.
- Measure along the film from the nearest bottom frame-number label, never from the picture: where the camera
  put the frame is not part of the edge print.
- Capital height and distance from the film edge only for a new mark or a clear mismatch.
- A phone photo is good to about ±0.3 mm. Change an existing value only when it is off by more than 1 mm.
- Text width and word spacing follow from NegPy's one edge font; match cap height and position, not width.

## 4. Decide the change

Read the comments in `LAYOUTS` first: a layout that already prints the structure needs only `edge.style`.

| The photo shows | Change |
|---|---|
| Different words or a short code | `edge.text` (printed product), `edge.maker`, `edge.code` (Fuji type code, KEYKODE letters) |
| A DX code | `edge.dx` as "part1-part2", as decoded in step 2. A cassette number from the Big Film Database is a fallback when nothing decodes; say so in `source` |
| No DX code where the layout prints one | a layout with `dx=()`; drop `edge.dx` |
| The structure of an existing layout | `edge.style` = that key; run `render_edge.py` to check |
| A structure no layout has, for this product line | a new `LAYOUTS` key named for the line ("kentmere", "kodak-alternating"), built from the existing runs |
| The same structure differs for every stock of a maker | edit that layout; render two other stocks that use it before and after |

`EdgeRun` fields are documented in `layouts.py`. Label runs take `every`/`phase` for every-Nth-frame marks;
`{batch}`, `{lot}` and `{roll}` are stand-in digits, because the real ones change per roll.

Every `edge` entry names its `source`: what was seen, for example `"photo of a Gold 200 roll: GB 200-7,
word order alternates; DX: Big Film Database"`. Leave a field out when nothing shows it. A sibling stock with
no photo (Pan 100 beside Pan 400) may take the same `style`, with its `source` saying so.

## 5. Verify

    uv run python .claude/skills/edge-print-from-photo/render_edge.py <stock-id> --out /tmp/after.png
    uv run pytest tests/test_film_edge.py tests/test_bundled_film_stocks.py tests/test_contact_sheet.py -q
    uv run ruff check negpy/services/film_edge tests/test_film_edge.py tests/test_bundled_film_stocks.py

Look at `/tmp/after.png` beside the photo. A new layout or mark gets a test in `tests/test_film_edge.py`;
a stock's data gets an assertion in `test_photographed_rolls_print_as_on_the_film` in
`tests/test_bundled_film_stocks.py`. After a DX or KEYKODE change, run `decode_barcodes.py` on the render.

## 6. Hand back

- The photo stays out of the repository. Name it in the commit message or the PR description.
- `docs/USER_GUIDE.md` describes the control, not stocks: a data-only change does not touch it.
- Report what the film showed, what changed, and what stays unverified (other formats, sibling stocks).

## Common mistakes

| Mistake | Fix |
|---|---|
| Text from memory, the box or the datasheet | Only what the film shows; else leave the field out |
| Keeping a DX number on a stock whose edge has none | Remove `edge.dx`; the layout carries `dx=()` |
| Editing a shared layout for one stock | New layout key, or data on the stock |
| Measuring from the frame picture | Measure from the frame-number labels |
| A per-stock note in USER_GUIDE | Commit message and `source` only |
| Changing 120 or half frame from a 35mm photo | Only the format in the photo |
