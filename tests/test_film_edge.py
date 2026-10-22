import re

import pytest

from negpy.features.metadata.gear_models import FilmEdge, FilmStock, GearLibrary
from negpy.services.film_edge.barcodes import _CODE128, DX_MODULE, KEYKODE_MODULE, code128c_bits, dx_bits
from negpy.services.film_edge.edge import (
    DX_START,
    Band,
    EdgeCode,
    EdgeStyle,
    EdgeText,
    ascii_upper,
    edge_family,
    edge_style,
    edge_items,
    label_text,
    parse_dx,
    split_maker,
)
from negpy.services.film_edge.geometry import PERF_PHASE, PERF_PITCH, SheetFormat, film_geometry
from negpy.services.film_edge.layouts import KEY_DOT, LAYOUTS, MARK_GAP

FULL = film_geometry(SheetFormat.FULL_FRAME)
HALF = film_geometry(SheetFormat.HALF_FRAME)


# Independent of the encoder: written from zxing-cpp's ODDXFilmEdgeReader layout.
def _decode_dx(clock, data):
    assert len(clock) == len(data) == 31
    assert list(clock[:5]) == [1] * 5 and list(clock[-3:]) == [1] * 3
    assert list(clock[5:28]) == [i % 2 for i in range(23)]
    assert list(data[:5]) == [1, 0, 1, 0, 1] and list(data[-3:]) == [1, 0, 1]
    payload = list(data[5:28])
    assert payload[0] == payload[8] == payload[20] == payload[22] == 0
    assert payload[21] == sum(payload[:21]) % 2

    def number(bits):
        return int("".join(str(b) for b in bits), 2)

    frame = number(payload[13:19])
    return number(payload[1:8]), number(payload[9:13]), f"{frame}{'A' if payload[19] else ''}"


def _dx(code):
    (_data_span, data), (_clock_span, clock) = code.tracks
    return clock, data


def test_dx_code_decodes_the_wikipedia_sample():
    # Wikipedia's DX encoding article captions an Agfa sample as decoding to 47-1/22A.
    assert _decode_dx(*dx_bits(22, True, (47, 1))) == (47, 1, "22A")


def test_dx_frame_field_wraps_at_64():
    assert _decode_dx(*dx_bits(64 + 5, False, (1, 0)))[2] == "5"


def test_dx_product_stand_in_is_stable_and_never_zero():
    style = EdgeStyle("kodak", "KODAK", "PORTRA 400", dx=True)
    assert style.dx_product == EdgeStyle("kodak", "KODAK", "PORTRA 400", dx=True).dx_product
    p1, p2 = style.dx_product
    assert 1 <= p1 <= 127 and 0 <= p2 <= 15


def test_label_sequence():
    assert [label_text(i) for i in range(6)] == ["1", "1A", "2", "2A", "3", "3A"]


def _bottom_labels(geometry, style, frames):
    return [(i.x, i.text) for i in edge_items(geometry, style, frames, 0, 1e9) if isinstance(i, EdgeText) and i.band == Band.BOTTOM]


def test_full_frame_numbers_sit_on_frames_and_a_numbers_in_the_gaps():
    labels = _bottom_labels(FULL, EdgeStyle("kodak", "KODAK", "TRI-X 400", dx=False), 3)
    assert labels == [(19.0, "1"), (38.0, "1A"), (57.0, "2"), (76.0, "2A"), (95.0, "3"), (114.0, "3A")]
    assert [FULL.frame_center(i) for i in range(3)] == [19.0, 57.0, 95.0]


def test_half_frames_carry_one_label_each():
    labels = _bottom_labels(HALF, EdgeStyle("ilford", "ILFORD", "HP5 PLUS", dx=False), 4)
    assert labels == [(HALF.frame_center(i), text) for i, text in enumerate(["1", "1A", "2", "2A"])]


def test_numbering_continues_across_strips():
    style = EdgeStyle("kodak", "KODAK", "PORTRA 400", dx=False)
    second_strip = [i for i in edge_items(FULL, style, 12, 6 * 38.0, 12 * 38.0) if isinstance(i, EdgeText) and i.band == Band.BOTTOM]
    inside = [i.text for i in second_strip if 6 * 38.0 < i.x < 12 * 38.0]
    on_cuts = [i.text for i in second_strip if i.x in (6 * 38.0, 12 * 38.0)]
    assert inside[0] == "7" and inside[-1] == "12"
    assert on_cuts == ["6A", "12A"]


def test_each_label_is_followed_by_its_own_code_clear_of_the_label():
    style = EdgeStyle("generic", "AGFA", "VISTA 200", dx=True)
    items = edge_items(FULL, style, 2, 0, 1e9)
    labels = [i for i in items if isinstance(i, EdgeText) and i.band == Band.BOTTOM]
    codes = [i for i in items if isinstance(i, EdgeCode)]
    assert len(codes) == len(labels) == 4
    for label, code in zip(labels, codes):
        assert label.align == "right"
        assert code.x - label.x >= 0.38
        assert code.x + code.length < label.x - DX_START + 19.0 - 2.5
        assert len(_dx(code)[1]) == 31 and code.length == 31 * DX_MODULE
    assert _decode_dx(*_dx(codes[1]))[2] == "1A"


def test_120_numbers_run_at_their_own_pitch():
    geometry = film_geometry(SheetFormat.MEDIUM, "6×7")
    style = EdgeStyle("ilford", "ILFORD", "HP5 PLUS", dx=False)
    numbers = [i for i in edge_items(geometry, style, 10, 0, 1e9) if isinstance(i, EdgeText) and i.band == Band.BOTTOM]
    pitches = {round(b.x - a.x, 3) for a, b in zip(numbers, numbers[1:])}
    assert pitches == {43.0}
    assert all(i.lead == "dots" for i in numbers)
    assert not any(isinstance(i, EdgeCode) for i in edge_items(geometry, style, 10, 0, 1e9))


def test_ilford_prints_its_name_every_frame_and_a_batch_every_third():
    style = EdgeStyle("ilford", "ILFORD", "HP5 PLUS", dx=True)
    top = [i.text for i in edge_items(FULL, style, 6, 0, 1e9) if isinstance(i, EdgeText) and i.band == Band.TOP]
    batch, lot = style.batch
    assert top.count("ILFORD HP5 PLUS") == 6
    assert top.count(batch) == top.count(f"-{lot}") == 2
    assert not set(top) & {str(n) for n in range(1, 7)}


def test_kentmere_prints_no_dx_code():
    items = edge_items(FULL, EdgeStyle("kentmere", "KENTMERE", "PAN 400", dx=True), 6, 0, 1e9)
    assert not any(isinstance(i, EdgeCode) for i in items)
    bottom = [i for i in items if isinstance(i, EdgeText) and i.band == Band.BOTTOM]
    assert [i.align for i in bottom][:2] == ["center", "center"]
    assert bottom[1].lead == "wedge" and bottom[1].x == pytest.approx(FULL.frame_center(0) + 19.0 + 1.0)
    name = next(i for i in items if isinstance(i, EdgeText) and i.text == "KENTMERE PAN 400")
    assert name.char_width > 0 and name.x == pytest.approx(FULL.frame_center(0) + 2.5)


def test_kodak_gold_swaps_its_word_order_every_frame():
    style = EdgeStyle("kodak-alternating", "KODAK", "GB 200-7")
    names = [i.text for i in edge_items(FULL, style, 4, 0, 1e9) if isinstance(i, EdgeText) and i.band == Band.TOP and "GB" in i.text]
    assert names == ["GB 200-7 KODAK", "KODAK GB 200-7", "GB 200-7 KODAK", "KODAK GB 200-7"]


def test_cine_film_prints_keykode_and_no_frame_numbers():
    style = EdgeStyle("cine", "EASTMAN", "5219", "EJ", dx=True)
    items = edge_items(FULL, style, 36, 0, 1e9)
    keys = [i for i in items if isinstance(i, EdgeText)]
    codes = [i for i in items if isinstance(i, EdgeCode) and i.module == KEYKODE_MODULE]
    index = sorted(i.x + i.length / 2 for i in items if isinstance(i, EdgeCode) and i.module != KEYKODE_MODULE)
    assert [k.text for k in keys[:2]] == [f"EJ {style.roll} 1000", f"EJ {style.roll} 1001"]
    assert all(re.fullmatch(r"EJ \d\d \d{4} \d{4}(\+32)?", k.text) and k.band == Band.BOTTOM and k.trail == "dot" for k in keys)
    assert len(codes) == len(keys)
    for key in keys:
        dot = key.x + MARK_GAP * key.cap + KEY_DOT / 2
        assert (dot - PERF_PHASE) / PERF_PITCH == pytest.approx(round((dot - PERF_PHASE) / PERF_PITCH))
        code = min(codes, key=lambda c: abs(c.x - dot))
        assert code.module == KEYKODE_MODULE and code.length == pytest.approx(23.432, abs=0.01)
        assert code.x + code.length == pytest.approx(dot + 6 * PERF_PITCH)
    plus = sorted(k.x for k in keys if k.text.endswith("+32"))
    full = sorted(k.x for k in keys if not k.text.endswith("+32"))
    assert full[1] - full[0] == pytest.approx(64 * PERF_PITCH) and plus and plus[0] > full[0]
    first_dot = full[0] + MARK_GAP * 1.52 + KEY_DOT / 2
    assert [round((x - first_dot) / PERF_PITCH) for x in index[:6]] == [8, 12, 16, 20, 24, 40]


def test_code_128_table_and_symbol_structure():
    assert len(_CODE128) == 107 and len(set(_CODE128)) == 107
    assert all(sum(map(int, widths)) == 11 for widths in _CODE128[:106]) and sum(map(int, _CODE128[106])) == 13
    bits = code128c_bits("0209076163436800")
    assert len(bits) == 123
    runs = [len(m.group()) for m in re.finditer(r"1+|0+", "".join(map(str, bits)))]
    symbols = ["".join(map(str, runs[i : i + 6])) for i in range(0, len(runs) - 7, 6)] + ["".join(map(str, runs[-7:]))]
    values = [_CODE128.index(sym) for sym in symbols]
    assert values[0] == 105 and values[-1] == 106
    assert values[1:-2] == [2, 9, 7, 61, 63, 43, 68, 0]
    assert values[-2] == (105 + sum(i * v for i, v in enumerate(values[1:-2], 1))) % 103


def test_edge_family():
    assert edge_family("Kodak Portra 400") == "kodak"
    assert edge_family("Ilford HP5 Plus") == "ilford"
    assert edge_family("Kentmere 400") == "kentmere"
    assert edge_family("Superia 400", "Fujifilm") == "fuji"
    assert edge_family("CineStill 800T") == "cine"
    assert edge_family("Kodak Vision3 250D") == "cine"
    assert edge_family("Foma Fomapan 400") == "generic"


def test_edge_text_is_plain_capitals():
    assert ascii_upper("Rollei  Ortho  25 Plus – Größe") == "ROLLEI ORTHO 25 PLUS GROE"


def test_frames_left_out_keep_the_other_frames_numbers():
    style = EdgeStyle("kodak", "KODAK", "GOLD 200", dx=True)
    full = [i.text for i in edge_items(FULL, style, 2, 0, 1e9, numbers=[0, 2]) if isinstance(i, EdgeText) and i.band == Band.BOTTOM]
    assert full == ["1", "1A", "3", "3A"]
    codes = [i for i in edge_items(FULL, style, 2, 0, 1e9, numbers=[0, 2]) if isinstance(i, EdgeCode)]
    assert _decode_dx(*_dx(codes[2]))[2] == "3"
    half = [
        i.text
        for i in edge_items(HALF, EdgeStyle("ilford", "ILFORD", "HP5 PLUS"), 2, 0, 1e9, numbers=[0, 3])
        if isinstance(i, EdgeText) and i.band == Band.BOTTOM
    ]
    assert half == ["1", "2A"]


def test_plain_film_prints_nothing_on_its_edges():
    style = EdgeStyle("kodak", "KODAK", "GOLD 200", dx=True, printed=False)
    assert edge_items(FULL, style, 6, 0, 1e9) == []
    assert edge_items(film_geometry(SheetFormat.MEDIUM, "6×6"), style, 6, 0, 1e9) == []


def test_every_layout_is_well_formed():
    fields = {"n": 1, "label": "1A", "maker": "M", "text": "T", "code": "C", "roll": "01 2345", "batch": "5048", "lot": "12"}
    for name, layout in LAYOUTS.items():
        assert set(layout.dx) <= {"color", "bw"}, name
        for fmt, runs in (("135", layout.runs_135), ("120", layout.runs_120)):
            for run in runs:
                run.text.format(**fields)
                assert run.on in (("number", "half", "pitch") if fmt == "135" else ("pitch",)), name
                assert run.on != "pitch" or run.pitch > 0, name
                assert run.every >= 1 and 0 <= run.phase < run.every, name
                assert {run.lead, run.trail, run.under} <= {"", "triangle", "wedge", "outline", "arrow", "dots", "dot"}, name
                assert run.barcode in ("", "keykode", "keykode+32", "index"), name


def test_split_maker_takes_a_leading_maker_word_off_the_name():
    assert split_maker("Kodak Portra 400") == ("KODAK", "PORTRA 400")
    assert split_maker("Portra 400", "Kodak") == ("KODAK", "PORTRA 400")
    assert split_maker("Fomapan 100 Classic", "Foma") == ("FOMA", "FOMAPAN 100 CLASSIC")
    assert split_maker("Velvia 50") == ("", "VELVIA 50")


def test_parse_dx_reads_only_numbers_the_edge_code_can_carry():
    assert parse_dx("95-13") == (95, 13)
    assert parse_dx(" 109 - 9 ") == (109, 9)
    for text in ("", "95", "0-1", "128-0", "95-16", "a-b", "1533"):
        assert parse_dx(text) is None, text


def test_a_real_dx_number_reaches_the_barcode():
    style = EdgeStyle("ilford", "ILFORD", "HP5 PLUS", dx=True, dx_number=(109, 9))
    code = next(i for i in edge_items(FULL, style, 1, 0, 1e9) if isinstance(i, EdgeCode))
    assert _decode_dx(*_dx(code)) == (109, 9, "1")


def test_catalog_edge_data_beats_the_name_guess():
    stock = FilmStock(manufacturer="Kodak", stock_name="Vision3 500T", edge=FilmEdge(style="cine", maker="EASTMAN", text="5219", code="EJ"))
    style = edge_style(stock)
    assert (style.layout, style.maker, style.text, style.code) == ("cine", "EASTMAN", "5219", "EJ")
    assert {i.text.split()[0] for i in edge_items(FULL, style, 12, 0, 1e9) if isinstance(i, EdgeText)} == {"EJ"}


def test_the_display_name_never_reaches_the_edge():
    stock = FilmStock(manufacturer="Kodak", stock_name="Ektacolor Pro 400", display_name="Kodak Ektacolor Pro 400 (formerly Portra 400)")
    assert (edge_style(stock).maker, edge_style(stock).text) == ("KODAK", "EKTACOLOR PRO 400")
    stock.edge = FilmEdge(text="PORTRA 400")
    assert (edge_style(stock).layout, edge_style(stock).text) == ("kodak", "PORTRA 400")


def test_a_personal_copy_reads_the_bundled_stock_of_the_same_name():
    bundled = FilmStock(manufacturer="Ilford", stock_name="HP5 Plus 400", edge=FilmEdge(dx="109-9"), is_bundled=True)
    personal = FilmStock(manufacturer="ilford", stock_name="hp5 plus 400")
    assert edge_style(personal, GearLibrary(film_stocks=[bundled, personal])).dx_number == (109, 9)
    assert edge_style(personal).dx_number is None


def test_a_respool_prints_its_source_stock():
    vision = FilmStock(id="v3", manufacturer="Kodak", stock_name="Vision3 500T", edge=FilmEdge(style="cine", maker="EASTMAN", text="5219"))
    respool = FilmStock(manufacturer="CineStill", stock_name="800T", edge=FilmEdge(prints_as="v3"))
    style = edge_style(respool, GearLibrary(film_stocks=[vision, respool]))
    assert (style.layout, style.maker, style.text) == ("cine", "EASTMAN", "5219")


def test_an_unknown_style_prints_the_generic_layout():
    assert edge_style(FilmStock(stock_name="X", edge=FilmEdge(style="nope"))).layout == "generic"


def test_a_free_text_film_name_guesses_the_style():
    style = edge_style(None, None, "Ilford HP5 Plus", "")
    assert (style.layout, style.maker, style.text) == ("ilford", "ILFORD", "HP5 PLUS")
