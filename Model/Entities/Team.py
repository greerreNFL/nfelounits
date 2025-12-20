'''
Team Class

Container for a team's six units (3 offensive + 3 defensive) plus QB tracking.
'''

from dataclasses import dataclass
from typing import Dict, Any, Tuple
from .Types import UnitType, Side
from .Unit import Unit
from .TeamQb import TeamQb


@dataclass
class Team:
    '''
    Represents a team with all offensive and defensive units plus QB tracking
    
    Team has 6 units + QB tracker:
    - Pass offense, Rush offense, ST offense
    - Pass defense, Rush defense, ST defense
    - QB tracker (expected QB value)
    '''
    team_abbr: str
    pass_off: Unit
    rush_off: Unit
    st_off: Unit
    pass_def: Unit
    rush_def: Unit
    st_def: Unit
    qb: TeamQb
    
    def __post_init__(self):
        '''Ensure all units and QB have correct team reference'''
        for unit in [self.pass_off, self.rush_off, self.st_off, self.pass_def, self.rush_def, self.st_def]:
            unit.team = self.team_abbr
        self.qb.team = self.team_abbr
    
    def get_units(self) -> Tuple[Unit, Unit, Unit, Unit, Unit, Unit]:
        '''Get all units for the team'''
        return self.pass_off, self.rush_off, self.st_off, self.pass_def, self.rush_def, self.st_def
        
    def get_total_off_value(self) -> float:
        '''Sum of all three unit offensive values'''
        return self.pass_off.value + self.rush_off.value + self.st_off.value
    
    def get_total_def_value(self) -> float:
        '''Sum of all three unit defensive values'''
        return self.pass_def.value + self.rush_def.value + self.st_def.value
    
    def as_record(self) -> Dict[str, Any]:
        '''Return team state as dictionary'''
        return {
            'team': self.team_abbr,
            'pass_off': round(self.pass_off.value, 3),
            'pass_def': round(self.pass_def.value, 3),
            'rush_off': round(self.rush_off.value, 3),
            'rush_def': round(self.rush_def.value, 3),
            'st_off': round(self.st_off.value, 3),
            'st_def': round(self.st_def.value, 3),
            'total_off': round(self.get_total_off_value(), 3),
            'total_def': round(self.get_total_def_value(), 3),
            'expected_qb_value': round(self.qb.expected_value, 3)
        }
