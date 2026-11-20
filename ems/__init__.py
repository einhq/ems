"""EMS — EiN Media Service. See docs/api.md for the contract."""

import json
from pathlib import Path

# The version of the API, which is EMS's release: version.json beside the package, raised by the
# daily publish as the commits since the last release call for (docs/api.md, Versioning).
API_VERSION: str = json.loads((Path(__file__).resolve().parents[1] / "version.json").read_text())["version"]
