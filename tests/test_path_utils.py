import unittest
from unittest.mock import patch

from utils.path_utils import ensure_display_path, normalize_local_path, windows_to_wsl_path, wsl_to_windows_path


class PathUtilsTests(unittest.TestCase):
    def test_windows_to_wsl_path(self) -> None:
        self.assertEqual(
            windows_to_wsl_path("C:\\Users\\86159\\Desktop\\test.jpg"),
            "/mnt/c/Users/86159/Desktop/test.jpg",
        )

    def test_wsl_to_windows_path(self) -> None:
        self.assertEqual(
            wsl_to_windows_path("/mnt/c/Users/86159/Desktop/test.jpg"),
            "C:\\Users\\86159\\Desktop\\test.jpg",
        )

    def test_normalize_local_path_preserves_absolute_path(self) -> None:
        with patch("utils.path_utils.os.name", "posix"):
            normalized = normalize_local_path("/mnt/c/Users/86159/Desktop/test.jpg")
        self.assertEqual(normalized, "/mnt/c/Users/86159/Desktop/test.jpg")

    def test_normalize_local_path_preserves_windows_path_on_native_windows(self) -> None:
        with patch("utils.path_utils.os.name", "nt"):
            normalized = normalize_local_path("C:/Users/86159/Desktop/test.jpg")
        self.assertEqual(normalized, "C:\\Users\\86159\\Desktop\\test.jpg")

    def test_normalize_local_path_converts_wsl_path_on_native_windows(self) -> None:
        with patch("utils.path_utils.os.name", "nt"):
            normalized = normalize_local_path("/mnt/c/Users/86159/Desktop/test.jpg")
        self.assertEqual(normalized, "C:\\Users\\86159\\Desktop\\test.jpg")

    def test_normalize_local_path_converts_windows_path_on_wsl(self) -> None:
        with patch("utils.path_utils.os.name", "posix"):
            normalized = normalize_local_path("C:\\Users\\86159\\Desktop\\test.jpg")
        self.assertEqual(normalized, "/mnt/c/Users/86159/Desktop/test.jpg")

    def test_ensure_display_path_returns_windows_style_for_wsl_mount(self) -> None:
        self.assertEqual(
            ensure_display_path("/mnt/c/Users/86159/Desktop/test.jpg"),
            "C:\\Users\\86159\\Desktop\\test.jpg",
        )


if __name__ == "__main__":
    unittest.main()
