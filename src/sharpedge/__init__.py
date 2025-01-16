# read version from installed package
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("sharpedge")
except PackageNotFoundError:
    __version__ = "0.0.0"  


