'''
UnitModelStateManager Class

Controller class handling IO and serialization for UnitModelState.
All serialization logic lives here - entity classes remain unchanged.
'''

import os
import json
from dataclasses import fields
from typing import Dict, List, Any, Optional

from .UnitModelState import UnitModelState
from ..Entities import (
    UnitType, Side, Pace, Unit,
    Team, TeamQb,
    LeagueBaseline, LeagueQb, LeaguePace,
    TeamGameRecord
)


class UnitModelStateManager:
    '''
    Controller class for managing UnitModel state persistence.
    
    Handles:
    - Serialization of entities to JSON-safe dictionaries
    - Deserialization from JSON back to entity objects
    - File IO with configurable paths
    
    Default save location: Model/State/model_state.json
    '''
    
    # Default location for state files (relative to package root)
    DEFAULT_STATE_DIR = os.path.join(os.path.dirname(__file__))
    DEFAULT_STATE_FILE = 'model_state.json'
    
    # Type registry for deserialization
    TYPE_REGISTRY = {
        'Pace': Pace,
        'Unit': Unit,
        'Team': Team,
        'TeamQb': TeamQb,
        'LeagueBaseline': LeagueBaseline,
        'LeagueQb': LeagueQb,
        'LeaguePace': LeaguePace,
        'UnitModelState': UnitModelState,
    }
    
    # Enum registry for deserialization
    ENUM_REGISTRY = {
        'UnitType': {'pass': UnitType.PASS, 'rush': UnitType.RUSH, 'st': UnitType.SPECIAL_TEAMS},
        'Side': {'off': Side.OFFENSE, 'def': Side.DEFENSE},
    }
    
    def __init__(self):
        '''Initialize state manager'''
        pass
    
    # ==================== Generic Serialization ====================
    
    def _serialize(self,
        obj: Any,
        exclude: set = {'params'}
    ) -> Any:
        '''
        Recursively serialize any object to JSON-safe format.
        
        Handles:
        - Dataclasses: Convert to dict with __type__ marker, recurse on fields
        - Dicts: Recurse on values
        - Lists: Recurse on items
        - Enums: Extract .value string
        - Primitives: Return as-is
        
        Parameters:
            obj: Object to serialize
            exclude: Field names to skip (default: {'params'})
        
        Returns:
            JSON-serializable representation
        '''
        if hasattr(obj, '__dataclass_fields__'):
            # Dataclass - convert to dict with type marker
            result = {'__type__': obj.__class__.__name__}
            for field in fields(obj):
                if field.name in exclude:
                    continue
                result[field.name] = self._serialize(getattr(obj, field.name), exclude)
            return result
        elif isinstance(obj, dict):
            return {k: self._serialize(v, exclude) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize(item, exclude) for item in obj]
        elif hasattr(obj, 'value'):
            # Enum - return string value
            return obj.value
        else:
            # Primitive (str, int, float, bool, None)
            return obj
    
    # ==================== Generic Deserialization ====================
    
    def _deserialize(self,
        data: Any,
        config: Dict[str, Any]
    ) -> Any:
        '''
        Recursively deserialize JSON data back to objects.
        
        Handles:
        - Dicts with __type__: Reconstruct as dataclass
        - Dicts without __type__: Recurse on values
        - Lists: Recurse on items
        - Primitives: Return as-is
        
        Parameters:
            data: JSON data to deserialize
            config: Model config (injected into entities that need params)
        
        Returns:
            Reconstructed object
        '''
        if isinstance(data, dict):
            if '__type__' in data:
                # Reconstruct dataclass
                type_name = data['__type__']
                cls = self.TYPE_REGISTRY[type_name]
                # Build kwargs, recursing on nested values
                kwargs = {}
                for key, val in data.items():
                    if key == '__type__':
                        continue
                    # Handle special enum fields
                    if type_name == 'Unit' and key == 'unit_type':
                        kwargs[key] = self.ENUM_REGISTRY['UnitType'][val]
                    elif type_name == 'Unit' and key == 'side':
                        kwargs[key] = self.ENUM_REGISTRY['Side'][val]
                    else:
                        kwargs[key] = self._deserialize(val, config)
                # Inject params for entities that need config
                if type_name in ('Unit', 'TeamQb', 'LeagueBaseline', 'LeagueQb', 'LeaguePace'):
                    kwargs['params'] = config
                return cls(**kwargs)
            else:
                # Plain dict - recurse on values
                return {k: self._deserialize(v, config) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._deserialize(item, config) for item in data]
        else:
            # Primitive
            return data
    
    # ==================== Public IO Methods ====================
    
    def _get_default_path(self) -> str:
        '''Get the default state file path.'''
        return os.path.join(self.DEFAULT_STATE_DIR, self.DEFAULT_STATE_FILE)
    
    def save(self,
        teams: Dict[str, Team],
        league_baseline: LeagueBaseline,
        league_qb: LeagueQb,
        league_pace: LeaguePace,
        team_game_records: List[TeamGameRecord],
        filepath: Optional[str] = None
    ) -> str:
        '''
        Save model state to a JSON file.
        
        Parameters:
            teams: Dictionary of team abbreviation -> Team object
            league_baseline: LeagueBaseline object
            league_qb: LeagueQb object
            team_game_records: List of TeamGameRecord dicts
            filepath: Path to save to. Defaults to Model/State/model_state.json
        
        Returns:
            The filepath where state was saved
        '''
        if filepath is None:
            filepath = self._get_default_path()
        # Determine season/week from most recent record
        if team_game_records:
            latest = team_game_records[-1]
            season = latest['season']
            week = latest['week']
        else:
            season = 0
            week = 0
        # Create state object
        state = UnitModelState(
            season=season,
            week=week,
            teams=teams,
            league_baseline=league_baseline,
            league_qb=league_qb,
            league_pace=league_pace,
            team_game_records=team_game_records,
        )
        # Serialize and write
        data = self._serialize(state)
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return filepath
    
    def load(self, config: Dict[str, Any], filepath: Optional[str] = None) -> UnitModelState:
        '''
        Load model state from a JSON file.
        
        Parameters:
            config: Model config dict (needed to reconstruct entities with params)
            filepath: Path to load from. Defaults to Model/State/model_state.json
        
        Returns:
            UnitModelState with reconstructed entity objects
        
        Raises:
            FileNotFoundError: If the state file doesn't exist
        '''
        if filepath is None:
            filepath = self._get_default_path()
        with open(filepath, 'r') as f:
            data = json.load(f)
        return self._deserialize(data, config)
    
    def exists(self, filepath: Optional[str] = None) -> bool:
        '''
        Check if a state file exists at the given path.
        
        Parameters:
            filepath: Path to check. Defaults to Model/State/model_state.json
        
        Returns:
            True if the file exists, False otherwise
        '''
        if filepath is None:
            filepath = self._get_default_path()
        return os.path.exists(filepath)
