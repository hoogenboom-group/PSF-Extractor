from pkg_resources import get_distribution

from . import parallel

from .extractor import *
from .gauss import *
from .plotting import *
from .util import *


__version__ = get_distribution('PSF-Extractor').version
__all__ = ['parallel']
