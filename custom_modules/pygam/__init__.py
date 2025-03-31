"""
GAM toolkit
"""

from __future__ import absolute_import

# Use relative imports instead of absolute ones.
from .pygam2 import GAM, LinearGAM, LogisticGAM, GammaGAM, PoissonGAM, InvGaussGAM, ExpectileGAM
from .terms import l, s, f, te, intercept

__all__ = [
    'GAM', 'LinearGAM', 'LogisticGAM', 'GammaGAM', 'PoissonGAM',
    'InvGaussGAM', 'ExpectileGAM', 'l', 's', 'f', 'te', 'intercept'
]

__version__ = '0.8.0'
