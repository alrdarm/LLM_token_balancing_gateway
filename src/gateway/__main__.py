"""Local entry point: ``python -m gateway`` or the ``gateway`` script."""

from __future__ import annotations

import uvicorn

from gateway.config import get_settings


def main() -> None:
    """Run the gateway with uvicorn using deployment settings."""
    settings = get_settings()
    uvicorn.run(
        "gateway.api.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_config=None,  # configure_logging owns formatting
    )


if __name__ == "__main__":
    main()
