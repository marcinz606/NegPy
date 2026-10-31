"""The bundled film stock list: every entry parses, labels read cleanly, and a roll folder named
the way people name them resolves to the right stock, or to none when the name fits a family."""

import pytest
from negpy.features.metadata.gear_models import FilmColorType, FilmStock, GearLibrary
from negpy.services.assets.gear import GearProfiles
from negpy.services.assets.gear_match import match_gear_for_folder
from negpy.services.film_edge.edge import edge_style, parse_dx
from negpy.services.film_edge.layouts import LAYOUTS


@pytest.fixture(scope="module")
def stocks() -> list[FilmStock]:
    return GearProfiles._read_bundled_list("film_stocks.json", FilmStock)


@pytest.fixture(scope="module")
def library(stocks) -> GearLibrary:
    return GearLibrary(cameras=[], lenses=[], film_stocks=stocks)


def test_every_bundled_stock_parses_with_a_known_type_a_speed_and_a_note(stocks):
    assert len(stocks) > 100
    for stock in stocks:
        assert stock.is_bundled
        assert stock.color_type is not FilmColorType.OTHER, stock.id
        assert stock.iso > 0, stock.id
        assert stock.notes, stock.id


def test_ids_display_names_and_maker_stock_pairs_are_unique(stocks):
    for key in (lambda s: s.id, lambda s: s.resolved_display_name, lambda s: (s.manufacturer, s.stock_name)):
        seen = [key(s) for s in stocks]
        assert len(seen) == len(set(seen)), [k for k in seen if seen.count(k) > 1]


def test_the_film_label_never_doubles_the_maker(stocks):
    # "Kentmere Pan 100", not "Kentmere Kentmere 100": the label reaches the exported metadata.
    for stock in stocks:
        words = stock.full_film_label.split()
        assert words[0] != words[1], stock.full_film_label


def test_edge_data_is_sourced_and_reads_as_edge_print(stocks, library):
    edged = [s for s in stocks if s.edge is not None]
    assert len(edged) > 40
    for stock in edged:
        edge = stock.edge
        assert edge.source, stock.id
        assert not edge.style or edge.style in LAYOUTS, stock.id
        assert not edge.dx or parse_dx(edge.dx), stock.id
        if edge.prints_as:
            source = library.get_film_stock(edge.prints_as)
            assert source is not None and source.edge is not None and not source.edge.prints_as, stock.id
        style = edge_style(stock, library)
        assert style.text.isascii() and style.text == style.text.upper(), stock.id


def test_a_renamed_kodak_stock_prints_its_legacy_mark(library):
    assert edge_style(library.get_film_stock("film-portra-400"), library).text == "PORTRA 400"
    assert edge_style(library.get_film_stock("film-ektapan-400"), library).text == "400TMY"


def test_photographed_rolls_print_as_on_the_film(library):
    def style(stock_id):
        return edge_style(library.get_film_stock(stock_id), library)

    assert (style("film-hp5-400").maker, style("film-hp5-400").text) == ("ILFORD", "HP5 PLUS")
    gold = style("film-gold-200")
    assert (gold.layout, gold.text, gold.dx_number) == ("kodak-alternating", "GB 200-7", (96, 12))
    assert style("film-kentmere-400").layout == "kentmere" and style("film-kentmere-400").dx_number is None
    assert [style(s).code for s in ("film-vision3-250d-ahu", "film-vision3-500t-ahu", "film-cinestill-bwxx")] == ["EN", "EJ", "KE"]
    # photographed rolls (edge-print-from-photo): the DX each shows on its rebate
    assert (style("film-fujifilm-400").layout, style("film-fujifilm-400").dx_number) == ("fuji", (95, 7))
    assert (style("film-lomography-cn-400").layout, style("film-lomography-cn-400").dx_number) == ("generic", (97, 9))
    assert (style("film-fujifilm-fujicolor-100").layout, style("film-fujifilm-fujicolor-100").dx_number) == ("fuji", (39, 1))
    assert [(c.at, c.color, c.thickness, c.dashed) for c in style("film-fujifilm-fujicolor-100").color_lines] == [
        (3.5, "green", 0.45, True),
        (2.6, "red", 0.22, False),
    ]
    assert (style("film-rollei-superpan-200").layout, style("film-rollei-superpan-200").dx_number) == ("generic", None)
    assert style("film-rollei-superpan-200").color_lines == ()


def test_superia_premium_400_prints_its_green_and_red_lines(library):
    style = edge_style(library.get_film_stock("film-fujifilm-superia-premium-400"), library)
    assert [(line.at, line.color, line.thickness, line.dashed) for line in style.color_lines] == [
        (2.6, "green", 0.22, False),
        (3.5, "red", 0.45, False),
    ]


@pytest.mark.parametrize(
    "folder, stock_id",
    [
        ("2026-05_portra400_wedding", "film-portra-400"),  # the pre-2026 name still resolves
        ("portra160", "film-portra-160"),
        ("portra800", "film-ektacolor-pro-800"),
        ("tmax100", "film-ektapan-100"),
        ("tmax400_street", "film-ektapan-400"),  # a subject word beside the stock does not cancel it
        ("p3200", "film-ektapan-p3200"),
        ("kentmere400", "film-kentmere-400"),  # the pre-2022 Kentmere name
        ("trix", "film-tri-x-400"),
        ("hp5", "film-hp5-400"),
        ("hp5 plus", "film-hp5-400"),  # "plus" is shared with four Ilford names; "hp5" carries the digit
        ("delta3200", "film-delta-3200"),
        ("acros", "film-fujifilm-neopan-acros-ii-100"),
        ("velvia50", "film-fujifilm-velvia-50"),
        ("fujifilm superia premium 400", "film-fujifilm-superia-premium-400"),  # a full stock name must resolve, so its color lines draw
        ("phoenix", "film-harman-phoenix-ii-200"),
        ("800t", "film-cinestill-800t"),
        ("bwxx", "film-cinestill-bwxx"),
        ("candido800", "film-candido-800-reserve"),
        ("nc500", "film-orwo-wolfen-nc500"),
        ("fomapan400", "film-foma-fomapan-400-action"),
        ("rpx400", "film-rollei-rpx-400"),
        ("blackbird", "film-rollei-blackbird-100"),
    ],
)
def test_a_folder_named_after_a_stock_resolves_to_it(library, folder, stock_id):
    assert match_gear_for_folder(folder, library).film_stock_id == stock_id


@pytest.mark.parametrize("folder", ["portra", "ektacolor pro", "superia 400", "vision3 500t", "kodak gold", "lucky trix"])
def test_a_family_word_or_two_stocks_in_one_name_resolve_to_nothing(library, folder):
    assert match_gear_for_folder(folder, library).film_stock_id == ""
