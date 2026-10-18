---
name: update-translations
description: Update NegPy's registered UI translations and localized panel guides after UI text changes, upstream merges or releases.
---

# Update translations

Keep the translated interface and panel guides aligned with the current English
sources. Paths below are relative to the repository root.

## Inspect the change

- Read the Translations section in `CONTRIBUTING.md` and the current diff. Use the
  requested comparison ref, when provided, to find changed UI text and controls.
  For a clean release checkout, review changes since the last translation update
  or previous release. `git log -- negpy/translations/zh_CN.json` locates catalog
  updates; an empty working-tree diff does not show which new controls lack marks.
- Read `LANGUAGES` in `negpy/kernel/system/i18n.py` for the registered locales.
  Honor any locale limit in the request; report unfinished entries in other locales.
- Find a message's call sites with `git grep -F 'tr("<text>"'`. Read them and the
  matching `docs/USER_GUIDE.md` panel before translating technical text.
  Existing catalogs and localized guides provide the terminology for each language.

## Mark display text

The extractor finds literal arguments to `tr`, `mark`, `count_of` and `plural`.
It cannot find an unmarked label or an already interpolated f-string. Inspect new
labels, tooltips, actions, status messages and dialogs in the diff before extraction.

Translate the complete template, then insert named values:

```python
tr("Could not open {name}: {error}", name=path.name, error=error)
```

- Use `mark("English label")` when the label is also a lookup key; call `tr` at
  display time. Keep catalog keys in English and match the source text exactly.
- Keep enum values, action IDs, settings keys, paths, file names and user text
  unchanged. A translated choice uses the original value in `itemData`/`data`.
  Read `currentData()` and select with `findData()`; the existing `add_choice_items`
  and `select_choice` helpers also work with `StepCombo`.
- Use `count_of` or `plural` from `negpy/kernel/system/text.py` for counts. An
  English `s` suffix does not apply to Chinese, Japanese or Korean.

## Refresh and translate catalogs

Run from the repository root:

```bash
make translations
```

This updates every registered JSON catalog, preserves exact-key translations,
removes obsolete keys and copies the English guide to
`negpy/translations/USER_GUIDE.en.md`. Review the diff before and after extraction.
A changed English key becomes a new entry: use its previous translation as context,
but check the changed meaning before reusing it.

Fill empty or outdated values in the requested catalogs. Edit JSON values through
a JSON parser or targeted patches so English keys remain intact. Keep UTF-8, the
extractor's indentation and the final newline.

- Preserve every named placeholder, repeated occurrence, conversion such as `!r`
  and format specifier such as `:.1f`.
- Preserve HTML tags and their order, link targets, commands, technical units and
  search syntax such as `film:portra iso:>=400`. Translate the surrounding prose.
- Keep terminology consistent across labels, tooltips, shortcuts, tutorials and
  guides. Check film and darkroom terms against their use in the English guide.
- Shared technical notation and product names can stay in English. Do not copy
  English prose into empty values merely to make the coverage check pass.

## Update panel guides

Update affected sections in `negpy/translations/USER_GUIDE.<locale>.md` from
`docs/USER_GUIDE.md`. Keep the same `<!-- panel:key src:hash -->` markers and a
heading immediately after each marker. `src` is the hash of the English section
the panel translates: `tests/test_i18n.py` lists each stale panel with its new
marker, so set it once the panel matches the English text. Add new panels, remove retired control descriptions
and preserve documented ranges, defaults and mode restrictions. The runtime falls
back to English per panel; that fallback does not mean the panel was translated.

Let `make translations` generate `USER_GUIDE.en.md`; edit its source in `docs/`.

## Validate

`make translations` can exit successfully while reporting missing translations.
Use the check command and tests to verify the result:

```bash
make format
make translations
make translations-check
```

For focused checks during an update:

```bash
uv run pytest tests/test_i18n.py tests/test_section_help_dialog.py tests/test_section_guides.py tests/test_step_combo.py
```

Before committing, `make all` provides the repository's required checks and
includes those tests.

The check command covers catalog keys and format signatures. The tests also check
HTML tags, display text outside `tr`, stale guide panels, guide coverage, Qt
translations and stored values in localized controls.
Review the final diff for changes to existing translations.

Language is selected before desktop imports; module constants are translated on
import, so changes require a restart. For visual review, use a fresh process for
each affected language, check the changed panels for clipped text and keep app data
in an absolute scratch directory, for example
`make run-sim NEGPY_USER_DIR=/absolute/scratch`.

Report the locales and panels updated, the checks run and any remaining missing
translations or English guide fallbacks.
