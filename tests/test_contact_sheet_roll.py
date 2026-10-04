from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import piexif
import pytest
import tifffile
from PIL import Image

from negpy.domain.models import ExportResolutionMode, WorkspaceConfig
from negpy.features.geometry.models import AspectRatio, AutocropMode
from negpy.features.metadata.gear_models import FilmColorType, FilmStock
from negpy.features.process.models import ProcessMode
from negpy.services.export import contact_sheet_roll as roll
from negpy.services.export.contact_sheet_edge import EdgeFamily
from negpy.services.export.contact_sheet_layout import SheetFormat, film_geometry
from negpy.services.export.contact_sheet_roll import (
    FrameFacts,
    SheetFrame,
    creation_order,
    cropped_aspect,
    infer_format,
    parse_frame_120,
    read_birth_time,
    read_capture_time,
    read_frame_facts,
    read_scan_size,
    roll_label_text,
    sheet_black,
    sheet_look,
    tile_params,
    turns_for,
    upright_aspect,
)


def _frame(name="a.jpg", half=0, capture="", birth=0.0, size=(3000, 2000), **meta) -> SheetFrame:
    config = WorkspaceConfig()
    if meta:
        config = replace(config, metadata=replace(config.metadata, **meta))
    asset = {"name": name, "path": f"/roll/{name}", "hash": name}
    if half:
        asset["half"] = half
    return SheetFrame(asset, config, FrameFacts(capture, birth, size))


class _Library:
    def __init__(self, stock):
        self.stock = stock

    def get_film_stock(self, stock_id):
        return self.stock if stock_id == self.stock.id else None


class TestOrder:
    def test_exif_time_when_every_frame_has_one(self):
        frames = [_frame("b.jpg", capture="2026-01-01 10:00:02", birth=1), _frame("a.jpg", capture="2026-01-01 10:00:01", birth=2)]
        assert [f.name for f in creation_order(frames)] == ["a.jpg", "b.jpg"]

    def test_file_creation_time_when_any_frame_lacks_exif(self):
        frames = [
            _frame("a.jpg", capture="2026-01-01 10:00:01", birth=30),
            _frame("b.jpg", capture="", birth=10),
            _frame("c.jpg", capture="2026-01-01 10:00:00", birth=20),
        ]
        assert [f.name for f in creation_order(frames)] == ["b.jpg", "c.jpg", "a.jpg"]

    def test_ties_break_by_name_then_half(self):
        frames = [_frame("x.tif", half=2, birth=5), _frame("x.tif", half=1, birth=5), _frame("a.tif", birth=5)]
        assert [(f.name, f.half) for f in creation_order(frames)] == [("a.tif", 0), ("x.tif", 1), ("x.tif", 2)]


class TestFileFacts:
    def test_jpeg_capture_time(self, tmp_path):
        path = str(tmp_path / "scan.jpg")
        exif = piexif.dump({"Exif": {piexif.ExifIFD.DateTimeOriginal: b"2026:08:11 14:03:07"}})
        Image.new("RGB", (30, 20)).save(path, exif=exif)
        assert read_capture_time(path) == "2026-08-11 14:03:07"

    def test_tiff_capture_time_from_its_header(self, tmp_path):
        path = str(tmp_path / "scan.tif")
        tifffile.imwrite(path, np.zeros((20, 30, 3), np.uint8), photometric="rgb", datetime="2026:08:11 14:03:08")
        assert read_capture_time(path) == "2026-08-11 14:03:08"

    def test_no_date_and_unreadable_files(self, tmp_path):
        path = str(tmp_path / "plain.png")
        Image.new("RGB", (30, 20)).save(path)
        assert read_capture_time(path) == ""
        assert read_capture_time(str(tmp_path / "missing.jpg")) == ""

    def test_birth_time_falls_back_to_mtime_where_the_os_has_none(self, monkeypatch):
        # Linux has no st_birthtime.
        monkeypatch.setattr(roll.os, "stat", lambda _path: SimpleNamespace(st_mtime=123.0))
        assert read_birth_time("/any") == 123.0

    def test_scan_size_follows_exif_orientation_and_the_half_split(self, tmp_path):
        path = str(tmp_path / "turned.jpg")
        exif = piexif.dump({"0th": {piexif.ImageIFD.Orientation: 6}})
        Image.new("RGB", (300, 200)).save(path, exif=exif)
        assert read_scan_size({"path": path}) == (200, 300)
        plain = str(tmp_path / "pair.jpg")
        Image.new("RGB", (360, 240)).save(plain)
        assert read_scan_size({"path": plain, "half": 1, "split_x": 0.5}) == (180, 240)

    def test_read_frame_facts(self, tmp_path):
        path = str(tmp_path / "f.jpg")
        Image.new("RGB", (300, 200)).save(path)
        facts = read_frame_facts({"path": path})
        assert facts.capture_time == "" and facts.birth_time > 0 and facts.scan_size == (300, 200)


class TestFormat:
    def test_metadata_majority(self):
        assert infer_format([_frame(format="35mm"), _frame(format="35mm"), _frame(format="120")]) == (SheetFormat.FULL_FRAME, "6×6")

    def test_other_naming_a_120_size(self):
        frames = [_frame(format="Other", format_other="6x7 Mamiya")] * 3
        assert infer_format(frames) == (SheetFormat.MEDIUM, "6×7")

    def test_120_size_from_the_crop_shape(self):
        frames = [_frame(format="120", size=(5600, 4480))] * 4
        assert infer_format(frames) == (SheetFormat.MEDIUM, "6×7")

    def test_split_halves_or_the_roll_mode_mean_half_frame(self):
        assert infer_format([_frame(half=1, format="35mm")])[0] == SheetFormat.HALF_FRAME
        assert infer_format([_frame(format="35mm")], half_frame_roll=True)[0] == SheetFormat.HALF_FRAME

    def test_no_metadata_reads_the_shape(self):
        assert infer_format([_frame(size=(4000, 4000))] * 3) == (SheetFormat.MEDIUM, "6×6")
        assert infer_format([_frame(size=(3000, 2000))] * 3)[0] == SheetFormat.FULL_FRAME
        # A 6x9 roll has the 35mm shape.
        assert infer_format([_frame(size=(8400, 5600))] * 3)[0] == SheetFormat.FULL_FRAME

    def test_parse_frame_120(self):
        assert parse_frame_120("6×4,5") == "6×4.5"
        assert parse_frame_120("645") == "6×4.5"
        assert parse_frame_120("6x12 pano") == "6×12"
        assert parse_frame_120("4×5") == ""


class TestOrientation:
    def test_upright_aspect_swaps_for_a_quarter_turn(self):
        frame = _frame(size=(3000, 2000))
        assert upright_aspect(frame) == pytest.approx(1.5)
        turned = replace(frame, config=replace(frame.config, geometry=replace(frame.config.geometry, rotation=1)))
        assert upright_aspect(turned) == pytest.approx(2 / 3)
        assert turns_for(turned, film_geometry(SheetFormat.FULL_FRAME)) == 3

    def test_a_portrait_crop_of_a_level_frame_stays_upright(self):
        frame = _frame(size=(3000, 2000))
        cropped = replace(frame, config=replace(frame.config, geometry=replace(frame.config.geometry, crop_rect=(0.3, 0.0, 0.7, 1.0))))
        assert cropped_aspect(cropped) == pytest.approx(1 / 0.6)
        assert turns_for(cropped, film_geometry(SheetFormat.FULL_FRAME)) == 0

    def test_tile_shape_stands_in_for_an_unknown_scan_size(self):
        frame = SheetFrame({"name": "s", "path": "/s", "hash": "s"}, WorkspaceConfig(), FrameFacts())
        assert upright_aspect(frame) is None
        assert upright_aspect(frame, (200, 300)) == pytest.approx(1.5)


class TestLook:
    def test_positive_scan_of_a_color_negative_stock_prints_ra4_with_dx(self):
        stock = FilmStock(manufacturer="Kodak", stock_name="Portra 400", color_type=FilmColorType.COLOR_NEGATIVE)
        frame = _frame(film_stock_id=stock.id, film="Kodak Portra 400")
        positive = replace(
            frame, config=replace(frame.config, process=replace(frame.config.process, process_mode=ProcessMode.E6, positive_source=True))
        )
        look = sheet_look([positive], SheetFormat.FULL_FRAME, "", _Library(stock))
        assert look.palette == "color"
        assert look.edge.dx and look.edge.family == EdgeFamily.KODAK
        assert look.edge.stock == "KODAK PORTRA 400"

    def test_slide_stock_prints_as_a_slide(self):
        frame = _frame(film="Kodak Ektachrome E100", film_color_type=FilmColorType.COLOR_SLIDE.value)
        look = sheet_look([frame], SheetFormat.FULL_FRAME)
        assert look.palette == "slide" and not look.edge.dx

    def test_process_mode_decides_when_the_film_is_unknown(self):
        frame = _frame()
        bw = replace(frame, config=replace(frame.config, process=replace(frame.config.process, process_mode=ProcessMode.BW)))
        assert sheet_look([bw], SheetFormat.FULL_FRAME).palette == "bw"
        assert sheet_look([frame], SheetFormat.FULL_FRAME).palette == "color"

    def test_ilford_black_and_white_carries_dx_and_120_never_does(self):
        frame = _frame(film="Ilford HP5 Plus", film_color_type=FilmColorType.BW_NEGATIVE.value)
        assert sheet_look([frame], SheetFormat.FULL_FRAME).edge.dx
        assert not sheet_look([frame], SheetFormat.MEDIUM).edge.dx
        tri_x = _frame(film="Kodak Tri-X 400", film_color_type=FilmColorType.BW_NEGATIVE.value)
        assert not sheet_look([tri_x], SheetFormat.FULL_FRAME).edge.dx

    def test_gear_display_name_avoids_a_doubled_maker(self):
        stock = FilmStock(manufacturer="Kentmere", stock_name="Kentmere 400", color_type=FilmColorType.BW_NEGATIVE)
        frame = _frame(film_stock_id=stock.id)
        assert sheet_look([frame], SheetFormat.FULL_FRAME, "", _Library(stock)).edge.stock == "KENTMERE 400"

    def test_sheet_black_follows_the_paper_model(self):
        assert sheet_black(False) == 0
        assert 14 <= sheet_black(True) <= 17
        frame = _frame()
        paper = replace(frame, config=replace(frame.config, exposure=replace(frame.config.exposure, paper_black=True)))
        assert sheet_look([paper, paper, frame], SheetFormat.FULL_FRAME).black == sheet_black(True)


def test_roll_label_text():
    frames = [
        _frame(
            film="Ilford HP5 Plus",
            film_iso=400,
            push_pull=1,
            developer="ID-11",
            process_dilution="1+1",
            camera_make="Nikon",
            camera_model="Nikon F3",
            capture_date="2026-08-11 14:00",
        )
    ] * 2
    assert roll_label_text("Roll 12", frames) == "Roll 12 · Ilford HP5 Plus @ EI 800 · ID-11 1+1 · Nikon F3 · 2026-08-11"
    assert roll_label_text("", [_frame()]) == ""


def test_tile_params_strip_the_print_layout():
    config = WorkspaceConfig()
    config = replace(
        config,
        finish=replace(config.finish, border_size=1.5, carrier_width=2.0),
        export=replace(config.export, export_resolution_mode=ExportResolutionMode.PRINT, paper_aspect_ratio=AspectRatio.R_5_4),
        geometry=replace(config.geometry, crop_from_auto=True, autocrop_mode=AutocropMode.FILM),
    )
    params = tile_params(config)
    assert params.finish.border_size == 0 and params.finish.carrier_width == 0
    assert params.export.export_resolution_mode == ExportResolutionMode.ORIGINAL
    assert params.export.paper_aspect_ratio == AspectRatio.ORIGINAL
    assert params.geometry.autocrop_mode == AutocropMode.IMAGE


def test_paths_of_a_composite():
    assert roll.asset_paths({"path": "/a", "hdr_paths": ("/a", "/b")}) == ["/a", "/b"]
    assert roll.asset_paths({"path": "/a"}) == ["/a"]
