'''
Data Module

Contains classes for loading, aggregating, and splitting NFL play-by-play data.
'''

from .DataLoader import DataLoader
from .DataSplitter import DataSplitter

__all__ = [
    'DataLoader',
    'DataSplitter'
]

