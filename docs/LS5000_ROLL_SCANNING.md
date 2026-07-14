# Nikon LS-5000 roll scanning

NegPy can read a whole roll in the Nikon LS-5000, show small previews, and scan the frames you choose. The saved TIFFs are linear negative masters. NegPy applies inversion and editing later, so a scan does not bake in a look.

## What you need

- A Nikon LS-5000 ED running firmware 1.03.
- An SA-30, or an adapter that reports the same 40-slot roll capacity.
- The patched `coolscan3` SANE driver. The black-and-white path depends on its frame, subframe, full-window, auto-exposure, and multisampling controls.
- Direct USB access on macOS or Linux.

The 40 slots are positions the feeder can address. They are not an exposure count. A 24 or 36 exposure roll can end with blank, partial, or otherwise unusable slots.

NegPy enables the roll controls when the scanner reports a live 40-slot capacity. If a parked feeder cannot report its capacity, you can explicitly enable the **SA-30 / converted SA-21 (40 slots)** profile. A live non-40 capacity always overrides that saved profile and keeps the roll controls unavailable.

## Preview and select a roll

1. Insert the film and open the Scan panel.
2. Choose the LS-5000 and click **Load Roll Thumbnails**. The feeder travels across the roll once and NegPy places a contact sheet in the main workspace.
3. Choose **Colour negative (C-41)** or **Black-and-white negative**.
4. Select the frames you want. Ctrl-click, Cmd-click, Shift-click, and drag selection all work.
5. Click **Scan selected**.

An orange `!` is an advisory warning. Blank or partial tail slots often get one because the feeder exposes positions beyond the last photographed frame. The slot stays selectable, but you should check the thumbnail.

A red `!` means that NegPy inferred the scanner position instead of finding a clear local frame gap. This often happens on slot 1 when the roll has a long clear leader. Select the slot, inspect the complete thumbnail, then check **I reviewed this inferred position**. The badge turns blue with a check mark. Changing the Film Spacing Offset or reloading the thumbnail clears that approval, so you must review the new crop.

The approval is not a bypass. It records the exact thumbnail, offset, scanner origin, and reviewed roll fingerprint. Before any full-quality C-41 frame starts, the capture worker reads a fresh low-resolution roll index and compares its film content and transport geometry with the reviewed preview. Normal scanner noise and small gain changes are allowed. A different, reordered, or stale roll is refused before the fine scan is armed.

The contact sheet uses its own display levels so that dense negatives and the orange mask remain readable. Those levels do not change scanner auto-exposure or the pixels written to the TIFF.

**Preview meter inset** controls which part of each thumbnail NegPy uses to calculate its display brightness. You can inset the meter by 0% to 30% from every edge; the default is 10%. At that setting, the calculation uses the center 80% of the image in each direction, which keeps a dark rebate from driving the preview levels. A translucent box on every thumbnail shows the area being metered. The full thumbnail, including the rebate, stays visible.

Changing the inset rerenders every loaded thumbnail immediately from the saved preview data. It does not move the film, run scanner auto-exposure again, or change the final scan.

**Preview display** defaults to an inverted positive so the contact sheet is easy to read. Turn on **Show non-inverted negative** to see a display-leveled negative instead. NegPy uses one shared RGB display range in this mode so the film base and channel balance remain visible. Switching modes rerenders the saved 97 dpi preview and does not contact the scanner.

The positive contact sheet is a quick linear inversion for choosing frames. It is not the full NegPy C-41 render. Full-quality scans remain scanner-linear negative masters; NegPy applies its C-41 color processing after import and can export the processed positive.

The line under the scanner name shows what the device is doing. It reports detection, preview loading, the current selected frame and slot, completion, and recovery errors. "Connected" means the scanner responded to the most recent detection or refresh. NegPy does not probe the USB device in the background while a scan owns it.

![NegPy LS-5000 full-roll contact sheet](images/ls5000-full-roll-preview.jpg)

## Fix a frame boundary

**Film Spacing Offset:** Use this when a thumbnail cuts into the photograph or includes too much of the next frame. The signed number and slider both start at zero. Negative values move the frame left and positive values move it right. With the contact sheet focused, `Alt+Left` and `Alt+Right` change the value one row at a time.

- Slot 1 accepts offsets from 0 to 144. Later slots accept -144 to 144.

Click **Reload Thumbnail** after changing the value. NegPy recrops that frame from the saved whole-roll preview and resolves the same offset through the scanner's transport table. An edited frame cannot be scanned until the matching thumbnail has returned. If the position was inferred, review and approve the reloaded thumbnail before scanning it.

If you eject or reinsert the film, or power-cycle the scanner, load the whole roll again. Frame coordinates belong to one insertion and must not be reused after the film moves.

## Full-quality capture

| Material | Scanner capture | Infrared | Import behavior |
| --- | --- | --- | --- |
| Colour negative (C-41) | 4000 dpi, 16-bit, RGB 4x | One scanner IR plane | C-41 mode, IR dust repair on |
| Conventional silver B&W | 4000 dpi, 16-bit, RGB 4x | Off | B&W mode, IR dust repair off |

Colour scans produce four files with the same base name:

- The RGB negative master: `.tif`
- The scanner IR plane: `_IR.tif`
- A mask that marks valid IR samples: `_IR_VALID.tif`
- A scan receipt: `_SCAN.json`

A selected colour batch also produces a `negpy-ls5000-batch-<session-id>.json` receipt beside the finished frames. Each frame receipt records its physical roll slot, Film Spacing Offset, reviewed roll identity, clipping telemetry, and a scene-dependent focus-detail score. A low focus score asks for a 100% visual check; it does not reject a smooth photograph as out of focus. The batch receipt uses the same session ID and records the single release after the last frame or a safe stop. It also identifies the exact request, capture worker, and packaged capture bundle that produced the frames.

The RGB and IR files come from one scanner traversal. NegPy averages four transferred RGB samples. The packed stream carries one IR plane. The scanner may combine IR samples in firmware, but the wire data does not prove that, so NegPy records its IR multisample semantics as unresolved.

Black-and-white scans produce one RGB TIFF. NegPy does not request IR, create an IR sidecar, or enable IR dust repair. Silver blocks infrared light, so an IR defect map cannot distinguish dust from the photograph. Chromogenic black-and-white films such as Ilford XP2 are different. Scan those as C-41 if you want IR dust removal.

Both routes use autofocus and hardware auto-exposure on the positioned frame. NegPy checks the requested geometry before capture and verifies the dimensions, bit depth, resolution, and transport-smear result before accepting the TIFF. Each frame's large packed capture file is scratch data. NegPy removes it after that frame's TIFF set has been verified and promoted, before it acknowledges the frame and lets the scanner begin the next one.

The roll picker estimates disk use before scanning. It budgets 256 MiB for each selected output set, the 591 MiB packed scratch file used by the active frame, and a 1 GiB working reserve. The Scan button stays disabled when the selected output filesystem does not have enough free space. This estimate is deliberately conservative because lossless TIFF compression can make a grainy scan larger than its uncompressed pixel count.

The roll controls show their two resolutions separately. Whole-roll thumbnails use 97 dpi. Every selected full-quality frame uses 4000 dpi, 16-bit, and a scanner-linear TIFF master.

The depth menu defaults to **16-bit (Best quality)**. Turning on **Archival RGB 4x + IR** selects 16-bit and locks the depth control until you turn Archival off. The packed LS-5000 stream supplies one aligned IR plane; its internal IR sampling remains unknown.

The **Single-frame DPI** and **Single-frame format** controls apply only to conventional single-frame scans. The resolution menu selects the highest value reported by the device and marks it **Best quality**. TIFF and DNG contain the same scanner-linear capture. The file format does not control inversion; use the Process panel for that.

Roll preview is a separate, short scanner operation. It reads the roll table and thumbnails, then releases the device before a full scan begins.

Selected colour frames share one direct scanner session. At the start of that session, NegPy rereads the low-resolution roll index and confirms that it matches the roll you reviewed. It then resolves every selected Film Spacing Offset and sends one combined Nikon frame table. The scanner reservation stays open while it captures the selected frames, so it does not close and travel across the roll again between frames. If the roll identity, an approval, or an offset is stale, the batch stops before the first full scan. A clean finish or safe stop sends one release at the end.

Selected black-and-white frames still use the SANE route, which opens one scanner session per frame.

NegPy asks macOS to prevent idle sleep while a preview or selected-frame batch is running. Keep a laptop lid open; software cannot override lid sleep, a manual shutdown, or loss of scanner power.

## Stopping and recovery

**Stop after current frame** lets the active scanner transaction finish, then prevents the next selected frame from starting. Frames completed before a stop or later error remain in the output folder and are imported into NegPy. After the preview is loaded again, **Select remaining** reselects only the physical slots that did not finish. Review their alignment before restarting the batch.

Finished roll files use slot-aware names by default, so a retry does not silently turn a physical slot into a new sequence number. C-41 captures also keep the authoritative slot in their `_SCAN.json` receipt if you use a custom filename template. The current B&W route writes only its slot-aware RGB TIFF.

If a USB failure leaves the feeder in an uncertain state, NegPy stops the queue and asks for a power cycle. Load the roll thumbnails again after recovery.

Use the **Eject** button beside the scanner selector when the device reports an eject control. A successful eject clears the contact sheet because its frame coordinates no longer belong to the film's current position.

The whole-roll preview tries to re-arm a parked feeder, but some parked states still time out. A successful preview validates the live startup frame table before it enables selected scans. If the preview times out, power-cycle the scanner, reinsert the film, and load the roll again. The SANE black-and-white route also needs an awake feeder.

After import, the Retouch panel reports whether IR repair was applied, skipped, unavailable, or still pending. A skipped repair includes its reason. The scanner-linear RGB and IR masters are kept unchanged either way.
