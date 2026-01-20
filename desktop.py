import os
from src.desktop.main import main

os.environ["NUMBA_THREADING_LAYER"] = "tbb"

if __name__ == "__main__":
    main()
