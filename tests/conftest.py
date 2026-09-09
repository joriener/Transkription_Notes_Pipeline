# =============================================================
#  Transkription_Notes_Pipeline - tests/conftest.py
#  Puts the project root on sys.path so test modules can
#  "import db", "import reporter", ... without each file repeating
#  the bootstrap.
#
#  The test files written before this existed still carry their own
#  sys.path.insert line. That is harmless (inserting the same path
#  twice changes nothing for imports) and they were deliberately left
#  untouched, so this file only removes boilerplate from new tests.
# =============================================================

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
