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
   git tag --sort=-v:refname        # newest tag first
   ```
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
   - Other emails: reuse the handle that person already got in earlier `docs/CHANGELOG.md` entries (`git log`/grep the file); ask the user if unknown.
   - Known map: Paul Glover=@paulglover, jboneng=@jboneng, Robin/light-sntchr=@light-sntchr, Henning Ullrich=@hullrich.
   - **`marcinz606` is the owner — never credit them.**

4. **Drop housekeeping commits** — no changelog line for: README/badge edits, changelog typo fixes, merge/revert/"rogue file" cleanups, pure internal log-noise fixes, version bumps.

5. **Check what's already documented** — the top `##` section may already hold a few entries. Merge into it; don't duplicate.

6. **Write one bullet per remaining commit** in the house style (below), grouped New → Change → Fix.

7. **Show the draft in the reply.** On approval, insert into `docs/CHANGELOG.md` under the target `## X.X.X` heading (create it if absent), merged with any entries already there.

## House style

(Established preference — see the user's changelog-style memory.)

- **One line per entry**, no explanatory paragraphs: `Prefix: **bold lead** — terse clause naming what was done.` Skip the failure-mode/root-cause narrative even on fixes. The long multi-sentence bullets in old sections are **not** the target.
- Prefix each with `New:` / `Change:` / `Fix:` (use `Change/Fix:` if genuinely both). Group in that order.
- **No PR/issue numbers** (`(#604)`) — user-facing notes, not a dev index.
- British **"colour"** in prose (NegPy naming standard).
- Plain, factual, non-salesy — it's an open-source project. No marketing framing, no dunking on other tools.
- Credit code/idea contributors with a trailing `@handle`; **never** credit whoever merely requested or reported it.

## Example (0.41.0, accepted)

```markdown
- New: **Manage Database…** — dialog to inspect and clear stored data; Clear Saved Edits or Reset Everything, both guarded. @linkmodo
- Change: **Bottom toolbar streamlined** — zoom slider removed, GPU/CPU toggle moved to the overflow menu, tooltips on every item; Before/After and Peak Flat are now mutually exclusive and survive rotate/flip. @linkmodo
- Fix: **Saved export destination restored on reopen** — the Folder mode no longer resets to "Subfolder of source". @paulglover
```

## Gotcha

`git log`/`git show` output is piped through the RTK hook, which **truncates
long output and garbles it**. If a log looks cut off, rerun with `rtk proxy git …`
to bypass the filter, or dump to a file and open it with the Read tool (Read is
unhooked). Trust exit codes over rendered text.
