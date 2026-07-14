"""MyTT: technical-analysis formulas for Python 3.12 and newer."""

from .core import *  # noqa: F403
from .core import __all__ as _core_all
from .extras import *  # noqa: F403
from .extras import __all__ as _extras_all
from .tdx import *  # noqa: F403
from .tdx import __all__ as _tdx_all

__version__ = "4.1.0"
__all__ = [*_core_all, *_extras_all, *_tdx_all, "__version__"]
