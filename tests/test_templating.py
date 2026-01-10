import datetime
from src.services.export.templating import FilenameTemplater


def test_filename_templater_basic() -> None:
    templater = FilenameTemplater()
    context = {
        "original_name": "IMG_1234",
        "mode": "C41",
        "fmt": "JPEG",
        "colorspace": "sRGB",
        "border": False,
    }
    pattern = "positive_{{ original_name }}"
    assert templater.render(pattern, context) == "positive_IMG_1234"


def test_filename_templater_complex() -> None:
    templater = FilenameTemplater()
    today = datetime.date.today().isoformat()
    context = {
        "original_name": "IMG_1234",
        "mode": "B&W",
        "fmt": "TIFF",
        "colorspace": "Adobe RGB",
        "border": True,
    }
    pattern = (
        "{{ date }}_{{ mode }}_{{ original_name }}{% if border %}_border{% endif %}"
    )
    expected = f"{today}_B&W_IMG_1234_border"
    assert templater.render(pattern, context) == expected


def test_filename_templater_fallback() -> None:
    templater = FilenameTemplater()
    context = {"original_name": "IMG_1234"}
    # Invalid Jinja2 syntax
    pattern = "positive_{{ original_name"
    assert templater.render(pattern, context) == "positive_IMG_1234"


def test_filename_templater_empty_result_fallback() -> None:
    templater = FilenameTemplater()
    context = {"original_name": "IMG_1234"}
    pattern = "{% if False %}nothing{% endif %}"
    assert templater.render(pattern, context) == "positive_IMG_1234"
