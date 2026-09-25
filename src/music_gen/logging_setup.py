import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once for scripts and services."""
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Third-party libraries are chatty at INFO.
    for noisy in ("httpx", "urllib3", "huggingface_hub", "diffusers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
