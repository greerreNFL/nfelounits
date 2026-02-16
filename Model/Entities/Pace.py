'''
Pace Class

Tracks play count pace using EWMA and provides smoothing factor discounts
for games with abnormal play counts.
'''

import math
from dataclasses import dataclass


@dataclass
class Pace:
    '''
    Tracks a unit's game-to-game play count pace (mean and variance)
    using exponentially weighted moving average.

    Used to discount smoothing factors when a game has an abnormal
    number of plays (e.g. blowouts, weather delays).
    '''
    mean: float
    var: float = 0.0

    def get_sf_discount(self, plays: int, threshold: float) -> float:
        '''
        Calculate smoothing factor discount (Lorentzian) based on pace deviation.

        Returns a multiplier in (0, 1] -- 1.0 means no discount,
        lower values mean the game's play count was abnormal.
        Threshold is the z-score at which the discount is 50%. For instance,
        if the threshold is 4, that means at 4 standard deviations, the SF would be reduced by
        50%. At 2 standard deviations, the SF would be reduced by about 20%. At 1 standard
        deviation, it would be practiaclly unchanged.

        Parameters:
        * plays: Number of plays observed in this game
        * threshold: Z-score threshold controlling discount steepness

        Returns:
        * Discount multiplier for the smoothing factor
        '''
        if self.var <= 0:
            return 1.0
        std = math.sqrt(self.var)
        z = abs(plays - self.mean) / std
        return 1 / (1 + (z / threshold) ** 2)

    def update(self, plays: int, sf: float) -> None:
        '''
        Update pace mean and variance with a new observation.

        Parameters:
        * plays: Number of plays observed in this game
        * sf: Smoothing factor for the EWMA update
        '''
        deviation = plays - self.mean
        self.mean = sf * plays + (1 - sf) * self.mean
        self.var = sf * (deviation ** 2) + (1 - sf) * self.var

    def regress(self, league_mean: float, league_var: float, reversion: float) -> None:
        '''
        Apply offseason regression toward league averages.

        Parameters:
        * league_mean: League average pace for this unit type
        * league_var: League average pace variance for this unit type
        * reversion: Reversion rate (0 = no reversion, 1 = full reversion)
        '''
        if reversion > 0:
            self.mean = (1 - reversion) * self.mean + reversion * league_mean
        self.var = league_var

    @classmethod
    def from_league(cls, league_mean: float, league_var: float) -> 'Pace':
        '''
        Factory to initialize pace from league-wide averages.

        Parameters:
        * league_mean: League average pace for this unit type
        * league_var: League average pace variance for this unit type

        Returns:
        * New Pace instance seeded with league values
        '''
        return cls(mean=league_mean, var=league_var if league_var is not None else 0.0)
