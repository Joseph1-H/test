import sys
from pathlib import Path

# Let tests import the shared fake servers and the package without installing it.
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
