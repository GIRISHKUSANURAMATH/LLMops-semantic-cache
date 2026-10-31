"""``python -m llmops_cache`` starts the API server using HOST/PORT from the environment."""

from __future__ import annotations

import uvicorn

from llmops_cache.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "llmops_cache.main:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
