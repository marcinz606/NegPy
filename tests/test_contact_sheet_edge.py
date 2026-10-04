from negpy.services.export.contact_sheet_edge import (
    DX_MODULE,
    DX_START,
    Band,
    EdgeCode,
    EdgeFamily,
    EdgeStyle,
    EdgeText,
    ascii_upper,
    dx_bits,
    edge_family,
    edge_items,
    label_text,
)
from negpy.services.export.contact_sheet_layout import SheetFormat, film_geometry

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


def test_dx_code_decodes_the_wikipedia_sample():
    # Wikipedia's DX encoding article captions an Agfa sample as decoding to 47-1/22A.
    assert _decode_dx(*dx_bits(22, True, (47, 1))) == (47, 1, "22A")


def test_dx_frame_field_wraps_at_64():
    assert _decode_dx(*dx_bits(64 + 5, False, (1, 0)))[2] == "5"


def test_dx_product_stand_in_is_stable_and_never_zero():
    style = EdgeStyle(EdgeFamily.KODAK, "KODAK PORTRA 400", True)
    assert style.dx_product == EdgeStyle(EdgeFamily.KODAK, "KODAK PORTRA 400", True).dx_product
    p1, p2 = style.dx_product
    assert 1 <= p1 <= 127 and 0 <= p2 <= 15


def test_label_sequence():
    assert [label_text(i) for i in range(6)] == ["1", "1A", "2", "2A", "3", "3A"]


def _bottom_labels(geometry, style, frames):
    return [(i.x, i.text) for i in edge_items(geometry, style, frames, 0, 1e9) if isinstance(i, EdgeText) and i.band == Band.BOTTOM]


def test_full_frame_numbers_sit_on_frames_and_a_numbers_in_the_gaps():
    labels = _bottom_labels(FULL, EdgeStyle(EdgeFamily.KODAK, "KODAK TRI-X 400", dx=False), 3)
    assert labels == [(19.0, "1"), (38.0, "1A"), (57.0, "2"), (76.0, "2A"), (95.0, "3"), (114.0, "3A")]
    assert [FULL.frame_center(i) for i in range(3)] == [19.0, 57.0, 95.0]


def test_half_frames_carry_one_label_each():
    labels = _bottom_labels(HALF, EdgeStyle(EdgeFamily.ILFORD, "ILFORD HP5 PLUS", dx=False), 4)
    assert labels == [(HALF.frame_center(i), text) for i, text in enumerate(["1", "1A", "2", "2A"])]


def test_numbering_continues_across_strips():
    style = EdgeStyle(EdgeFamily.KODAK, "KODAK PORTRA 400", dx=False)
    second_strip = [i for i in edge_items(FULL, style, 12, 6 * 38.0, 12 * 38.0) if isinstance(i, EdgeText) and i.band == Band.BOTTOM]
    inside = [i.text for i in second_strip if 6 * 38.0 < i.x < 12 * 38.0]
    on_cuts = [i.text for i in second_strip if i.x in (6 * 38.0, 12 * 38.0)]
    assert inside[0] == "7" and inside[-1] == "12"
    assert on_cuts == ["6A", "12A"]


def test_each_label_is_followed_by_its_own_code_clear_of_the_label():
    style = EdgeStyle(EdgeFamily.GENERIC, "AGFA VISTA 200", dx=True)
    items = edge_items(FULL, style, 2, 0, 1e9)
    labels = [i for i in items if isinstance(i, EdgeText) and i.band == Band.BOTTOM]
    codes = [i for i in items if isinstance(i, EdgeCode)]
    assert len(codes) == len(labels) == 4
    for label, code in zip(labels, codes):
        assert label.align == "right"
        assert code.x - label.x >= 0.38
        assert code.x + code.length < label.x - DX_START + 19.0 - 2.5
        assert len(code.data) == 31 and code.length == 31 * DX_MODULE
    assert _decode_dx(codes[1].clock, codes[1].data)[2] == "1A"


def test_120_numbers_run_at_their_own_pitch():
    geometry = film_geometry(SheetFormat.MEDIUM, "6×7")
    style = EdgeStyle(EdgeFamily.ILFORD, "ILFORD HP5 PLUS", dx=False)
    numbers = [i for i in edge_items(geometry, style, 10, 0, 1e9) if isinstance(i, EdgeText) and i.band == Band.BOTTOM]
    pitches = {round(b.x - a.x, 3) for a, b in zip(numbers, numbers[1:])}
    assert pitches == {43.0}
    assert all(i.lead == "dots" for i in numbers)
    assert not any(isinstance(i, EdgeCode) for i in edge_items(geometry, style, 10, 0, 1e9))


def test_ilford_top_band_prints_the_maker_and_product_and_no_numbers():
    style = EdgeStyle(EdgeFamily.ILFORD, "KENTMERE 400", dx=True)
    top = [i.text for i in edge_items(FULL, style, 6, 0, 1e9) if isinstance(i, EdgeText) and i.band == Band.TOP]
    assert "KENTMERE" in top and "400" in top
    assert not any(t.isdigit() and t != "400" for t in top)


def test_cine_film_has_no_frame_numbers_and_no_code():
    style = EdgeStyle(EdgeFamily.CINE, "KODAK VISION3 500T", dx=False)
    items = edge_items(FULL, style, 6, 0, 1e9)
    assert all(isinstance(i, EdgeText) and i.band == Band.TOP and i.text == "KODAK VISION3 500T" for i in items)


def test_edge_family():
    assert edge_family("Kodak Portra 400") == EdgeFamily.KODAK
    assert edge_family("Ilford HP5 Plus") == EdgeFamily.ILFORD
    assert edge_family("Kentmere 400") == EdgeFamily.ILFORD
    assert edge_family("Superia 400", "Fujifilm") == EdgeFamily.FUJI
    assert edge_family("CineStill 800T") == EdgeFamily.CINE
    assert edge_family("Kodak Vision3 250D") == EdgeFamily.CINE
    assert edge_family("Foma Fomapan 400") == EdgeFamily.GENERIC


def test_edge_text_is_plain_capitals():
    assert ascii_upper("Rollei  Ortho  25 Plus – Größe") == "ROLLEI ORTHO 25 PLUS GROE"


def test_frames_left_out_keep_the_other_frames_numbers():
    style = EdgeStyle(EdgeFamily.KODAK, "KODAK GOLD 200", dx=True)
    full = [i.text for i in edge_items(FULL, style, 2, 0, 1e9, numbers=[0, 2]) if isinstance(i, EdgeText) and i.band == Band.BOTTOM]
    assert full == ["1", "1A", "3", "3A"]
    codes = [i for i in edge_items(FULL, style, 2, 0, 1e9, numbers=[0, 2]) if isinstance(i, EdgeCode)]
    assert _decode_dx(codes[2].clock, codes[2].data)[2] == "3"
    half = [
        i.text
        for i in edge_items(HALF, EdgeStyle(EdgeFamily.ILFORD, "ILFORD HP5 PLUS"), 2, 0, 1e9, numbers=[0, 3])
        if isinstance(i, EdgeText) and i.band == Band.BOTTOM
    ]
    assert half == ["1", "2A"]


def test_plain_film_prints_nothing_on_its_edges():
    style = EdgeStyle(EdgeFamily.KODAK, "KODAK GOLD 200", dx=True, printed=False)
    assert edge_items(FULL, style, 6, 0, 1e9) == []
    assert edge_items(film_geometry(SheetFormat.MEDIUM, "6×6"), style, 6, 0, 1e9) == []
