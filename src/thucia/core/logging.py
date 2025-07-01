import logging


def enable_logging(level=logging.INFO):
    """
    Enable logging with the specified level.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("prefect").setLevel(logging.WARNING)  # Reduce Prefect noise
    logging.getLogger("httpx").setLevel(logging.WARNING)  # Reduce HTTPX noise
