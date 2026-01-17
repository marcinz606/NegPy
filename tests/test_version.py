import unittest
from unittest.mock import patch, MagicMock
from src.kernel.system.version import check_for_updates
import json


class TestVersionCheck(unittest.TestCase):
    @patch("src.kernel.system.version.get_app_version")
    @patch("urllib.request.urlopen")
    def test_check_for_updates_available(self, mock_urlopen, mock_get_version):
        mock_get_version.return_value = "0.9.0"

        # Mock GitHub response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps({"tag_name": "v0.9.5"}).encode()
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        new_ver = check_for_updates()
        self.assertEqual(new_ver, "0.9.5")

    @patch("src.kernel.system.version.get_app_version")
    @patch("urllib.request.urlopen")
    def test_check_for_updates_latest(self, mock_urlopen, mock_get_version):
        mock_get_version.return_value = "0.9.5"

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps({"tag_name": "v0.9.5"}).encode()
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        new_ver = check_for_updates()
        self.assertIsNone(new_ver)

    @patch("src.kernel.system.version.get_app_version")
    @patch("urllib.request.urlopen")
    def test_check_for_updates_error(self, mock_urlopen, mock_get_version):
        mock_get_version.return_value = "0.9.0"
        mock_urlopen.side_effect = Exception("Network error")

        new_ver = check_for_updates()
        self.assertIsNone(new_ver)


if __name__ == "__main__":
    unittest.main()
