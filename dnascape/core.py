"""Compatibility API layer for DNAscape.

This module now re-exports the public API from focused submodules.
"""

import numpy as np
import pandas as pd

from .constants import *
from .io import *
from .utils import *
from .plotting import *
from .mapping import *
from .simulation import *
from .fitting import *
from .routing import *
