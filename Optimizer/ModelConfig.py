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
    def values(self) -> Dict[str, Any]:
        '''
        Get nested dictionary of parameter names to values
        
        Returns nested structure like:
        {
            'unit_config': {param_name: value, ...},
            'elo_config': {param_name: value, ...}
        }
        '''
        result = {}
        for key, param in self.params.items():
            ## split on first dot to get section and param name ##
            if '.' in key:
                section, param_name = key.split('.', 1)
                if section not in result:
                    result[section] = {}
                result[section][param_name] = param.value
            else:
                ## fallback for non-nested params ##
                result[key] = param.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        '''
        Create ModelConfig from dictionary
        
        Handles nested structure (section: {param_name: {value, description, ...}})
        Flattens params with dot notation (e.g., 'unit_config.pass_hfa_share')
        '''
        params = {}
        
        for section_key, section_value in data.items():
            if isinstance(section_value, dict):
                ## check if this is a section (nested dict with param objects) ##
                first_item_key = next(iter(section_value.keys()), None)
                if first_item_key and isinstance(section_value[first_item_key], dict) and 'value' in section_value[first_item_key]:
                    ## this is a config section like 'unit_config' or 'elo_config' ##
                    for param_name, param_data in section_value.items():
                        flattened_key = f'{section_key}.{param_name}'
                        params[flattened_key] = ModelParam.from_dict(param_data)
                else:
                    ## old flat format - single param ##
                    params[section_key] = ModelParam.from_dict(section_value)
            else:
                ## old format - create ModelParam with defaults ##
                params[section_key] = ModelParam(
                    value=section_value,
                    description=f'Parameter {section_key}',
                    opti_min=0.0,
                    opti_max=1.0 if 'share' in section_key or 'reversion' in section_key else 0.5
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
        Save ModelConfig to JSON file in nested structure
        
        Parameters:
        * filepath: Path to config file (defaults to package config.json)
        '''
        if filepath is None:
            package_folder = pathlib.Path(__file__).parent.parent.resolve()
            filepath = f'{package_folder}/config.json'
        
        ## reconstruct nested structure ##
        nested_data = {}
        for key, param in self.params.items():
            if '.' in key:
                section, param_name = key.split('.', 1)
                if section not in nested_data:
                    nested_data[section] = {}
                nested_data[section][param_name] = param.to_dict()
            else:
                ## fallback for non-nested params ##
                nested_data[key] = param.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(nested_data, f, indent=4)
    
    def update_config(self, updates: Dict[str, float]) -> None:
        '''
        Update parameter values
        
        Parameters:
        * updates: Dictionary of parameter_name -> new_value
        '''
        for key, value in updates.items():
            if key in self.params:
                self.params[key].value = value

