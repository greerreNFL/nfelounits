'''
RecalibrationObject Class

Simple dataclass holding recalibration data for blending ideal values.

Note: This is in its own file to avoid circular imports.
Unit.py needs RecalibrationObject for type hints, and UnitRecalibrator.py needs Unit.
Keeping RecalibrationObject separate breaks the cycle.
'''

from dataclasses import dataclass


@dataclass
class RecalibrationObject:
    '''
    Holds recalibration data for a single unit
    
    Attributes:
    * value: Ideal end-of-chain value from optimized priors
    * weight: Blend weight from sigmoid activation (0-1)
    '''
    value: float
    weight: float
