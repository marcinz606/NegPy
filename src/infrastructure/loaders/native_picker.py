import os
from typing import Optional
import subprocess
import sys
import json
from typing import List, Tuple
from src.core.io.interfaces import IFilePicker
from src.logging_config import get_logger

logger = get_logger(__name__)


class NativeFilePicker(IFilePicker):
    """
    OS-native file and folder picker.
    Calls the main entry point with specific flags to execute dialogs
    without starting the full Streamlit app.
    """

    def _run_subtask(self, flag: str, initial_dir: Optional[str] = None) -> str:
        """Runs the app as a subtask and returns stdout."""
        try:
            # Construct command based on execution environment
            if getattr(sys, "frozen", False):
                cmd = [sys.executable, flag]
            else:
                app_path = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "../../../app.py")
                )
                cmd = [sys.executable, app_path, flag]

            if initial_dir:
                cmd.append(initial_dir)

            logger.info(f"Launching native dialog subtask: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            if stderr:
                logger.debug(f"Subtask stderr: {stderr}")

            return stdout

        except subprocess.CalledProcessError as e:
            logger.error(f"Native dialog process failed: {e.stderr}")
            raise RuntimeError(f"Native dialog failed: {e.stderr}") from e

    def pick_files(self, initial_dir: Optional[str] = None) -> List[str]:
        """Opens a multi-file selection dialog via app subtask."""
        output = self._run_subtask("--pick-files", initial_dir)
        if not output:
            return []

        # Search for JSON array in the output buffer
        for line in reversed(output.splitlines()):
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                try:
                    paths = list(json.loads(line))
                    logger.info(f"Native picker returned {len(paths)} files")
                    return paths
                except json.JSONDecodeError:
                    continue

        logger.warning(
            "Native picker finished but no valid JSON paths were found in output."
        )
        return []

    def pick_folder(self, initial_dir: Optional[str] = None) -> Tuple[str, List[str]]:
        """Opens a directory selection dialog via app subtask."""
        output = self._run_subtask("--pick-folder", initial_dir)
        if not output:
            return "", []

        for line in reversed(output.splitlines()):
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                try:
                    res = json.loads(line)
                    root_path = str(res[0])
                    files = list(res[1])
                    logger.info(
                        f"Native folder picker returned {len(files)} files from {root_path}"
                    )
                    return root_path, files
                except (json.JSONDecodeError, IndexError, TypeError):
                    continue

        return "", []
