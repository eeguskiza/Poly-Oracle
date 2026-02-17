import sys
from pathlib import Path

from loguru import logger


def setup_logging(log_level: str = "DEBUG", log_dir: Path = Path("db/logs")) -> None:
    logger.remove()

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "poly-oracle.log"

    stderr_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{module}</cyan> | "
        "{message}"
    )

    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{module} | "
        "{message}"
    )

    logger.add(
        sys.stderr,
        format=stderr_format,
        level=log_level,
        colorize=True,
    )

    logger.add(
        log_file,
        format=file_format,
        level=log_level,
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        colorize=False,
    )

    logger.info(f"Logging initialized at {log_level} level")
