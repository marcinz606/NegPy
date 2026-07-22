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
    try:
        faulthandler.enable()
    except Exception:
        pass
    # Frozen scanner helpers and the offline packaging smoke must not import
    # the Qt desktop. Keep this dispatch before the GUI module import.
    from negpy.desktop.frozen_entry import dispatch_frozen_auxiliary

    if not dispatch_frozen_auxiliary(sys.argv):
        from negpy.desktop.main import main

        main()
