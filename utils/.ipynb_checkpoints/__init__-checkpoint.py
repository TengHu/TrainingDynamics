"""Useful utils
"""
from .misc import *
from .logger import *
from .eval import *


# progress bar
import os, sys
from progress.bar import Bar as Bar
sys.path.append(os.path.join(os.path.dirname(__file__), "progress"))
