---
name: changelog
description: "Use when asked to write, draft, or update the changelog for a NegPy release — e.g. \"changelog for 0.42.0\", \"draft release notes since the last tag\", \"what changed since 0.41.0\". Pulls every commit since the previous git tag, summarizes them into concise user-facing bullets, credits outside contributors by @handle, and drafts the entry into docs/CHANGELOG.md for approval. Keywords: changelog, release notes, CHANGELOG.md, version bump, since tag, PRs, contributors."
---

# Draft a release changelog

## Overview

Turn the commits since the last release tag into a `docs/CHANGELOG.md` entry:
concise user-facing bullets, contributors credited, in the house style.

**Core principle:** always draft first and show the user — apply to
`docs/CHANGELOG.md` only after they accept. The user asked for a draft to
review, not an automatic edit.

## Workflow

1. **Get the target version** from the request (e.g. `0.42.0`) and find the range:
   ```bash
   git tag --sort=-v:refname | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' | head -3
   ```
   Release tags are bare (`0.63.0`, no `v`); old `v0.0.x-alpha` tags sort first without the filter.
   - Target is **not yet a tag** (drafting an unreleased version, the usual case) → range is `<newest-tag>..HEAD`.
   - Target **is** an existing tag → range is `<tag-immediately-before-it>..<target-tag>`.

2. **List commits with authors + emails** (email carries the GitHub handle):
   ```bash
   git log <range> --format='%h|%an|%ae|%s'
   ```
   For a commit whose body matters (root cause, what actually changed), read it:
   ```bash
   git show -s --format='%s%n%b' <hash>
   ```

3. **Derive contributor handles** from the author email:
   - `NNN+handle@users.noreply.github.com` → `@handle` (e.g. `175231748+linkmodo@...` → `@linkmodo`).
   - Other emails: reuse the handle that person already got in earlier `docs/CHANGELOG.md` entries (`git log`/grep the file).
   - Still unknown: the PR author is the handle — `git log --merges --ancestry-path <hash>..HEAD --format=%s | grep 'pull request' | tail -1` names `from <handle>/<branch>`, or `gh pr list --state merged --search "<words>" --json number,author`. The commit name can differ from the PR author (Kiran Shila → @activexray). Ask the user only if both fail.
   - Known map: Paul Glover=@paulglover, jboneng=@jboneng, Robin/light-sntchr=@light-sntchr, Henning Ullrich=@hullrich, Jon Destoppeleire=@jondestoppeleire, Kurt Windisch=@kurtwindisch, Sean Harding=@seanharding, red_prs=@red-prs, Kiran Shila=@activexray.
   - **`marcinz606` / Marcin Zawalski is the owner — never credit them.** A bullet that is mostly the owner's work gets **no tag**, even when a contributor added a small part of it. When a theme mixes a contributor's substantial work with the owner's, split it so the contributor's part keeps its tag (Sabattier @jondestoppeleire apart from the owner's Shadow Hold).

4. **Drop housekeeping commits** — no changelog line for: README/badge edits, changelog typo fixes, merge/revert/"rogue file" cleanups, pure internal log-noise fixes, version bumps.

5. **One bullet per feature, never one per commit.** A feature built over
   several commits — the feature, then its fixes, follow-ups and polish — gets a
   single bullet describing the shipped state. Fold anything the user should
   know into that bullet and write no extra `Fix:` or `Change:` line: a bug that
   never reached a build is not news. Group the range's commits by feature
   first, then write. Check every candidate bullet against the range's other
   bullets *and* the entries already in the section, not only the new commits.

   The same holds for a fix to an existing panel or control touched more than
   once in the range: one bullet naming the end state, not a line per commit.

   Then **group by theme**: related features share one bullet (Analysis,
   Camera scanning, Export, Workflow, Catalogs), and fixes group by area
   (Scanning, File loading, GPU and CPU match). A bullet may carry several
   contributor handles. Put the release's headline feature first. A large
   release lands near 15–20 bullets, not one per feature.

6. **Check what's already documented** — the top `##` section may already hold a few entries. Merge into it; don't duplicate.

7. **Write one bullet per theme** in the house style (below), grouped New → Change → Fix.

8. **Show the draft in the reply.** On approval, insert into `docs/CHANGELOG.md` under the target `## X.X.X` heading (create it if absent), merged with any entries already there.

9. **Set `VERSION` to the requested version** — the repo-root `VERSION` file holds
   the bare number (`0.56.0`, no `v`, trailing newline). Write it in the same
   change as the accepted changelog entry.

10. **Release screenshot** — if the user prepped a new `docs/media/NNNN.png`
    (`git status` shows it untracked, the old one deleted), point `README.md`'s
    main-window image at it and commit both in the same change.

11. **Commit and PR** — on a `release/X.Y.Z` branch, commit
    `chore(release): X.Y.Z changelog, version and screenshot` (stage files by
    name), push, and open the PR only when the user asks.

## House style

(Established preference — see the user's changelog-style memory.)

- **One entry per theme**, not per commit or per feature — the shipped state, with its in-cycle fixes folded in.
- **One line per entry**, no explanatory paragraphs: `Prefix: **bold lead** — terse clause naming what was done.` Skip the failure-mode/root-cause narrative even on fixes. The long multi-sentence bullets in old sections are **not** the target.
- Prefix each with `New:` / `Change:` / `Fix:` (use `Change/Fix:` if genuinely both). Group in that order.
- **No PR/issue numbers** (`(#604)`) — user-facing notes, not a dev index.
- US **"color"** in prose, matching the UI labels (Color Negative, colour spellings were swept out in 0.50.0).
- Plain, factual, non-salesy — it's an open-source project. No marketing framing, no dunking on other tools.
- Credit code/idea contributors with trailing `@handle`s; **never** credit whoever merely requested or reported it, and never tag a bullet that is mostly the owner's work.

## Example (0.64.0, accepted)

```markdown
- New: **Expanded Analysis** — Waveform and Vectorscope views beside the Curve chart, a Clipping Overlay, zone strip hover that highlights the zone's pixels, a double-click on the zone strip toggles the Zone Overlay, and the step wedge prints through alt process and toning.
- New: **Camera scanning** — live view histogram and clipping zebras, Meter sets the shutter from one RAW probe, single-capture narrowband scanning that saves a sensor profile, a Method choice for the sensor calibration against blown blues, and a New Roll button. @jondestoppeleire @seanharding @thetalkingdrum
- Change: **Auto Crop** — a roll setting on the Crop card with Frame/Roll scope, mirrored on Geometry; Roll asks before a push changes it for other frames, and Square Before Crop runs the Auto Skew fit first. @kurtwindisch @jondestoppeleire
- Fix: **File loading** — palette, MinIsWhite, CMYK, planar and two-sample gray TIFFs, LogLuv, CR3/RAF orientation, and NegPy's own ICC profiles on reload.
```

## Gotcha

`git log`/`git show` output is piped through the RTK hook, which **truncates
long output and garbles it**. If a log looks cut off, rerun with `rtk proxy git …`
to bypass the filter, or dump to a file and open it with the Read tool (Read is
unhooked). Trust exit codes over rendered text.
