import sys
import io
import faulthandler
from src.desktop.main import main


def init_streams():
    """Fallback for None stdout/stderr in frozen Windows GUI."""

    class DummyStream(io.TextIOBase):
        def write(self, x):
            return len(x)

        def flush(self):
            pass

    if sys.stdout is None:
        sys.stdout = DummyStream()
    if sys.stderr is None:
        sys.stderr = DummyStream()


if __name__ == "__main__":
    init_streams()
    try:
        faulthandler.enable()
    except Exception:
        pass
    main()
