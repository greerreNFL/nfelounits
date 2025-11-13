'''
ModelConfig and ModelParam Classes

Data structures for managing model configuration parameters with optimization bounds.
'''

from dataclasses import dataclass, field
from typing import Dict, Any
import json
import pathlib


@dataclass
class ModelParam:
    '''
    Represents a single model parameter with optimization bounds
    '''
    value: float
    description: str
    opti_min: float
    opti_max: float
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelParam':
        '''Create ModelParam from dictionary'''
        return cls(
            value=data['value'],
            description=data.get('description', ''),
            opti_min=data.get('opti_min', 0.0),
            opti_max=data.get('opti_max', 1.0)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        '''Convert ModelParam to dictionary'''
        return {
            'value': self.value,
            'description': self.description,
            'opti_min': self.opti_min,
            'opti_max': self.opti_max
        }


@dataclass
class ModelConfig:
    '''
    Container for all model configuration parameters
    '''
    params: Dict[str, ModelParam] = field(default_factory=dict)
    
    @property
    def values(self) -> Dict[str, float]:
        '''Get dictionary of parameter names to values'''
        return {k: v.value for k, v in self.params.items()}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        '''
        Create ModelConfig from dictionary
        
        Handles both old format (param_name: value) and new format (param_name: {value, description, ...})
        '''
        params = {}
        for key, value in data.items():
            if isinstance(value, dict):
                ## new format ##
                params[key] = ModelParam.from_dict(value)
            else:
                ## old format - create ModelParam with defaults ##
                params[key] = ModelParam(
                    value=value,
                    description=f'Parameter {key}',
                    opti_min=0.0,
                    opti_max=1.0 if 'share' in key or 'reversion' in key else 0.5
                )
        return cls(params=params)
    
    @classmethod
    def from_file(cls, filepath: str = None) -> 'ModelConfig':
        '''
        Load ModelConfig from JSON file
        
        Parameters:
        * filepath: Path to config file (defaults to package config.json)
        '''
        if filepath is None:
            package_folder = pathlib.Path(__file__).parent.parent.resolve()
            filepath = f'{package_folder}/config.json'
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_file(self, filepath: str = None) -> None:
        '''
        Save ModelConfig to JSON file
        
        Parameters:
        * filepath: Path to config file (defaults to package config.json)
        '''
        if filepath is None:
            package_folder = pathlib.Path(__file__).parent.parent.resolve()
            filepath = f'{package_folder}/config.json'
        data = {k: v.to_dict() for k, v in self.params.items()}
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
    
    def update_config(self, updates: Dict[str, float]) -> None:
        '''
        Update parameter values
        
        Parameters:
        * updates: Dictionary of parameter_name -> new_value
        '''
        for key, value in updates.items():
            if key in self.params:
                self.params[key].value = value

