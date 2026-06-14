"""Free-hosting entrypoint for the Streamlit app.

Streamlit Community Cloud and Hugging Face Spaces commonly look for an app at
the repository root. The main app implementation lives in the package's
`main` section.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deep_researcher.main.streamlit_app import main


if __name__ == "__main__":
    main()
