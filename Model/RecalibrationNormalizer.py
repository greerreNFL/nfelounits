'''
RecalibrationNormalizer Class
Normalizes recalibration values to match the scale of UnitModel values.
'''
from typing import Dict, Any

class RecalibrationNormalizer:
    '''
    Normalizes recalibration values to prevent overfit extreme values
    from dominating the blend with UnitModel values.
    For each unit type, computes: m(week) = slope * week + intercept
    Then normalizes: normalized_value = m * recal_value
    '''
    UNIT_TYPES = ['pass_off', 'pass_def', 'rush_off', 'rush_def', 'st_off', 'st_def']
    
    def __init__(self, config: Dict[str, Any]):
        '''
        Initialize normalizer from config.
        Parameters:
        * config: Model config containing recal_normalizer section (already flattened)
        '''
        normalizer_config = config['recal_normalizer']
        self.coefficients: Dict[str, Dict[str, float]] = {}
        for unit_type in self.UNIT_TYPES:
            self.coefficients[unit_type] = {
                'm': normalizer_config[f'{unit_type}_m'],
                'b': normalizer_config[f'{unit_type}_b']
            }
    
    def get_normalization_factor(self, unit_type: str, week: int) -> float:
        '''
        Get the normalization factor for a given unit type and week.
        Returns: m(week) = slope * week + intercept
        '''
        coef = self.coefficients[unit_type]
        return coef['m'] * week + coef['b']
    
    def normalize(self, recal_value: float, unit_type: str, week: int) -> float:
        '''
        Normalize a recalibration value.
        Returns: normalized_value = m(week) * recal_value
        '''
        m = self.get_normalization_factor(unit_type, week)
        return m * recal_value

