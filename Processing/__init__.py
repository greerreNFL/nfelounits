'''
Processing Module

Classes for processing team game records into output files.
'''

from .BaseProcessor import BaseProcessor
from .UnitTeamsProcessor import UnitTeamsProcessor
from .NormalizationProcessor import UnitTeamsNormalizationProcessor
from .ValueCreatedProcessor import ValueCreatedProcessor
from .OpponentFacedProcessor import OpponentFacedProcessor
from .Utils import create_week_index, calculate_window, calculate_all_windows, forward_fill_weeks, normalize

__all__ = [
    'BaseProcessor',
    'UnitTeamsProcessor',
    'UnitTeamsNormalizationProcessor',
    'ValueCreatedProcessor',
    'OpponentFacedProcessor',
    ## utils ##
    'create_week_index',
    'calculate_window',
    'calculate_all_windows',
    'forward_fill_weeks',
    'normalize',
]

