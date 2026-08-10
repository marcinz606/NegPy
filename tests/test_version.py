import json
import unittest
from unittest.mock import MagicMock, patch

from negpy.kernel.system.version import fetch_latest_release, is_newer, parse_version


def _response(payload: dict, status: int = 200) -> MagicMock:
    response = MagicMock()
    response.status = status
    response.read.return_value = json.dumps(payload).encode()
    response.__enter__.return_value = response
    return response


class TestVersionParsing(unittest.TestCase):
    def test_parse_version_keeps_numeric_parts(self):
        self.assertEqual(parse_version("0.9.5"), [0, 9, 5])
        self.assertEqual(parse_version("v0.9.5"), [9, 5])  # the caller strips the "v"

    def test_a_later_release_is_newer(self):
        self.assertTrue(is_newer("0.9.5", "0.9.0"))
        self.assertTrue(is_newer("0.10.0", "0.9.9"))

    def test_the_same_or_older_release_is_not_newer(self):
        self.assertFalse(is_newer("0.9.5", "0.9.5"))
        self.assertFalse(is_newer("0.9.0", "0.9.5"))

    def test_an_unusable_version_is_never_newer(self):
        self.assertFalse(is_newer("", "0.9.0"))
        self.assertFalse(is_newer("0.9.5", "unknown"))


class TestFetchLatestRelease(unittest.TestCase):
    @patch("urllib.request.urlopen")
    def test_returns_the_payload(self, mock_urlopen):
        mock_urlopen.return_value = _response({"tag_name": "v0.9.5"})

        self.assertEqual(fetch_latest_release(), {"tag_name": "v0.9.5"})

    @patch("urllib.request.urlopen")
    def test_a_non_200_reads_as_no_release(self, mock_urlopen):
        mock_urlopen.return_value = _response({"tag_name": "v0.9.5"}, status=404)

        self.assertIsNone(fetch_latest_release())

    @patch("urllib.request.urlopen")
    def test_a_network_error_reads_as_no_release(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Network error")

        self.assertIsNone(fetch_latest_release())


if __name__ == "__main__":
    unittest.main()
