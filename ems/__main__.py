"""`python -m ems` — serve on EMS_HOST:EMS_PORT."""

import uvicorn

from .app import create_app
from .config import Settings


def main() -> None:
    settings = Settings.from_env()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="info", timeout_keep_alive=75)


if __name__ == "__main__":
    main()
