"""DocSpectra: schema-driven document parsing — any file in, spectral JSON out."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('docspectra')
except PackageNotFoundError:
    __version__ = '0.1.0'
