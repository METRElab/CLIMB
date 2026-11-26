"""Cerebellar testing platform: run granule-layer models through a fixed reference
cerebellar circuit on eyeblink conditioning, and compare two models live."""

from .granulelayer import GranuleLayer, ParamError

__version__ = '0.2.0'
__all__ = ['GranuleLayer', 'ParamError', '__version__']
