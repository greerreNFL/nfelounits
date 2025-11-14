'''
nfelounits - NFL Unit Performance Ratings

A Python package that decomposes team performance into measurable unit ratings
(QB/Passing, Rushing, Special Teams) using EPA (Expected Points Added) from 
play-by-play data. Uses EWMA (Exponentially Weighted Moving Average) updates 
with offseason regression.
'''

__version__ = '0.2.0'

## import main classes for easy access ##
from .Data import DataLoader, DataSplitter
from .Model import UnitType, Unit, Team, UnitModel, GameContext
from .Performance import UnitGrader
from .Optimizer import ModelConfig, ModelParam, ConfigOptimizer

__all__ = [
    ## data classes ##
    'DataLoader',
    'DataSplitter',
    ## model classes ##
    'UnitType',
    'Unit',
    'Team',
    'UnitModel',
    'GameContext',
    ## performance classes ##
    'UnitGrader',
    ## optimizer classes ##
    'ModelConfig',
    'ModelParam',
    'ConfigOptimizer',
]
