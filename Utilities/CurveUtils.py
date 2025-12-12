'''
Curve Utilities

Mathematical curve functions for adjustments and transformations.
'''

from typing import Optional


def s_curve(
    height: float,
    mp: float,
    x: float,
    direction: str = 'down',
    steepness: Optional[float] = None
) -> float:
    '''
    Calculate an s-curve for discounting or ramping values

    Parameters:
    * height: The maximum value of the curve
    * mp: The midpoint of the curve
    * x: The x-value to calculate the curve for
    * direction: The direction of the curve, either 'down' or 'up'
    * steepness: Optional steepness factor (default = 10/mp for backward compatibility)
    
    Returns:
    * The calculated s-curve value
    '''
    ## use default steepness if not provided ##
    if steepness is None:
        steepness = 10 / mp
    if direction == 'down':
        return (
            1 - (1 / (1 + 1.5 ** (
                (-1 * (x - mp)) *
                steepness
            )))
        ) * height
    else:
        return (1-(
            1 - (1 / (1 + 1.5 ** (
                (-1 * (x - mp)) *
                steepness
            )))
        )) * height
