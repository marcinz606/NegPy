import asyncio
import logging
from src.frontend.main import main
from src.logging_config import setup_logging

if __name__ == "__main__":
    setup_logging(level=logging.INFO)
    asyncio.run(main())