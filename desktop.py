import os
from src.desktop.main import main

os.environ["NUMBA_NUM_THREADS"] = "1"

if __name__ == "__main__":
    main()
