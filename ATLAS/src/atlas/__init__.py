"""ATLAS: Automated Tasks, Logic and Software. A local-first AI coding agent."""

import logging

__version__ = "0.1.0"

# Log only where setup_logging() points; never spill tracebacks onto the terminal.
logging.getLogger("atlas").addHandler(logging.NullHandler())
