import sys
import io
import faulthandler


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
    from negpy.desktop.startup import prepare_user_directory

    if not prepare_user_directory():
        sys.exit(1)
    from negpy.desktop.main import main

    try:
        faulthandler.enable()
    except Exception:
        pass
    main()
