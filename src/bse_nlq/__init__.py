"""Natural Language Query agent over the BSE ticketing database."""

from importlib.metadata import PackageNotFoundError, version

from bse_nlq.agent import NLQAgent, NLQResult, Outcome

try:
    __version__ = version("bse-nlq")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+local"

__all__ = ["NLQAgent", "NLQResult", "Outcome", "__version__"]
