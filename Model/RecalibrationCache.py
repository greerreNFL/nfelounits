'''
RecalibrationCache Classes

Persistence and orchestration for pre-computed recalibration values.
'''

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
import pandas as pd


@dataclass
class RecalibrationRecord:
    '''
    A single recalibration lookup entry.
    
    Attributes:
    * season: Season year
    * for_week: The week this value is FOR (predicting)
    * team: Team abbreviation
    * unit_type: Unit type ('pass_off', 'rush_def', etc.)
    * ideal_value: The optimized end-of-chain value
    '''
    season: int
    for_week: int
    team: str
    unit_type: str
    ideal_value: float


class RecalibrationSet:
    '''
    Collection of RecalibrationRecords with persistence.
    
    Provides efficient lookup by (season, for_week, team, unit_type).
    '''
    
    def __init__(self, records: Optional[List[RecalibrationRecord]] = None):
        '''
        Initialize with optional list of records.
        
        Parameters:
        * records: Optional list of RecalibrationRecord objects
        '''
        self.records: List[RecalibrationRecord] = records or []
        self._index: Dict[Tuple[int, int, str, str], int] = {}
        self._rebuild_index()
    
    def _rebuild_index(self) -> None:
        '''Rebuild the lookup index from records'''
        self._index = {}
        for i, record in enumerate(self.records):
            key = (record.season, record.for_week, record.team, record.unit_type)
            self._index[key] = i
    
    @classmethod
    def from_csv(cls, filepath: str) -> 'RecalibrationSet':
        '''
        Load RecalibrationSet from CSV file.
        
        Parameters:
        * filepath: Path to CSV file
        
        Returns:
        * RecalibrationSet populated from file
        '''
        if not os.path.exists(filepath):
            return cls([])
        df = pd.read_csv(filepath)
        records = [
            RecalibrationRecord(
                season=int(row['season']),
                for_week=int(row['for_week']),
                team=row['team'],
                unit_type=row['unit_type'],
                ideal_value=float(row['ideal_value'])
            )
            for _, row in df.iterrows()
        ]
        return cls(records)
    
    def to_csv(self, filepath: str) -> None:
        '''
        Save RecalibrationSet to CSV file.
        
        Parameters:
        * filepath: Path to save CSV
        '''
        data = [
            {
                'season': r.season,
                'for_week': r.for_week,
                'team': r.team,
                'unit_type': r.unit_type,
                'ideal_value': r.ideal_value
            }
            for r in self.records
        ]
        df = pd.DataFrame(data)
        ## sort for consistent output ##
        df = df.sort_values(['season', 'for_week', 'team', 'unit_type'])
        df.to_csv(filepath, index=False)
    
    def get(self, season: int, for_week: int, team: str, unit_type: str) -> Optional[RecalibrationRecord]:
        '''
        Lookup a specific record.
        
        Parameters:
        * season: Season year
        * for_week: Week the value is FOR
        * team: Team abbreviation
        * unit_type: Unit type
        
        Returns:
        * RecalibrationRecord if found, None otherwise
        '''
        key = (season, for_week, team, unit_type)
        idx = self._index.get(key)
        if idx is not None:
            return self.records[idx]
        return None
    
    def get_existing_weeks(self) -> Set[Tuple[int, int]]:
        '''
        Get set of (season, for_week) tuples that exist in the set.
        
        Returns:
        * Set of (season, for_week) tuples
        '''
        return {(r.season, r.for_week) for r in self.records}
    
    def get_missing_weeks(self, games: pd.DataFrame, min_week: int = 4) -> List[Tuple[int, int]]:
        '''
        Find (season, week) pairs in games that are missing from this set.
        
        Parameters:
        * games: Games DataFrame with 'season' and 'week' columns
        * min_week: Minimum week to consider (recalibration doesn't start until this week)
        
        Returns:
        * List of (season, for_week) tuples that need computation
        '''
        existing = self.get_existing_weeks()
        ## get unique (season, week) pairs from games ##
        all_weeks = set(games[['season', 'week']].drop_duplicates().itertuples(index=False, name=None))
        ## filter to weeks >= min_week ##
        all_weeks = {(s, w) for s, w in all_weeks if w >= min_week}
        ## return missing as sorted list ##
        missing = all_weeks - existing
        return sorted(missing)
    
    def upsert(self, records: List[RecalibrationRecord]) -> None:
        '''
        Add or update records.
        
        Parameters:
        * records: List of RecalibrationRecord objects to add/update
        '''
        for record in records:
            key = (record.season, record.for_week, record.team, record.unit_type)
            if key in self._index:
                ## update existing ##
                self.records[self._index[key]] = record
            else:
                ## add new ##
                self._index[key] = len(self.records)
                self.records.append(record)


class RecalibrationManager:
    '''
    Orchestrates recalibration computation and persistence.
    
    Manages the workflow of:
    1. Running UnitModel with state collection
    2. Running UnitRecalibrator for each captured state
    3. Persisting results to CSV
    '''
    
    DEFAULT_PATH = 'Output/recalibration_values.csv'
    
    def __init__(self,
        games: pd.DataFrame,
        config: Dict[str, Any],
        filepath: Optional[str] = None,
        min_week: int = 1
    ):
        '''
        Initialize manager, loading existing cache if present.
        
        Parameters:
        * games: Full games DataFrame
        * config: Model configuration
        * filepath: Path to cache file (default: Output/recalibration_values.csv)
        * min_week: Minimum week for recalibration (default: 4)
        '''
        self.games = games
        self.config = config
        self.filepath = filepath or self.DEFAULT_PATH
        self.min_week = min_week
        ## load existing set ##
        self.recal_set = RecalibrationSet.from_csv(self.filepath)
    
    def generate_all(self, verbose: bool = True) -> None:
        '''
        Generate recalibration values for all weeks.
        
        Runs UnitModel with StateCollector, then UnitRecalibrator
        for each captured state. Replaces any existing values.
        
        Parameters:
        * verbose: Whether to print progress
        '''
        from .UnitModel import UnitModel
        from .UnitRecalibrator import UnitRecalibrator
        from .UnitModelState import UnitModelStateCollector
        
        ## run model with state collection ##
        if verbose:
            print("Running UnitModel with state collection...")
        collector = UnitModelStateCollector()
        model = UnitModel(
            games=self.games,
            config=self.config,
            state_collector=collector
        )
        model.run()
        
        ## generate recalibration for each state ##
        states = collector.get_all_states()
        if verbose:
            print(f"Collected {len(states)} week states, generating recalibration values...")
        
        all_records: List[RecalibrationRecord] = []
        for state in states:
            if state.for_week < self.min_week:
                continue
            
            ## create recalibrator for this state ##
            recalibrator = UnitRecalibrator(
                games=self.games,
                config=self.config,
                league_baseline=state.league_baseline,
                league_qb=state.league_qb,
                season=state.season,
                week=state.for_week,
                teams=state.teams
            )
            
            ## extract ideal values for all teams/units ##
            if recalibrator.model is not None:
                for team_abbr in recalibrator.model.teams.keys():
                    for unit_type in ['pass_off', 'pass_def', 'rush_off', 'rush_def', 'st_off', 'st_def']:
                        ideal_value = recalibrator.get_ideal_value(team_abbr, unit_type)
                        if ideal_value is not None:
                            all_records.append(RecalibrationRecord(
                                season=state.season,
                                for_week=state.for_week,
                                team=team_abbr,
                                unit_type=unit_type,
                                ideal_value=ideal_value
                            ))
        
        ## replace set with new records ##
        self.recal_set = RecalibrationSet(all_records)
        if verbose:
            print(f"Generated {len(all_records)} recalibration records")
    
    def update(self, verbose: bool = True) -> None:
        '''
        Update recalibration values for missing weeks only.
        
        Runs UnitModel with StateCollector, identifies missing weeks,
        then runs UnitRecalibrator only for those weeks.
        
        Parameters:
        * verbose: Whether to print progress
        '''
        from .UnitModel import UnitModel
        from .UnitRecalibrator import UnitRecalibrator
        from .UnitModelState import UnitModelStateCollector
        
        ## find missing weeks ##
        missing_weeks = self.recal_set.get_missing_weeks(self.games, self.min_week)
        if not missing_weeks:
            if verbose:
                print("No missing weeks to compute")
            return
        
        if verbose:
            print(f"Found {len(missing_weeks)} missing weeks, running model...")
        
        ## run model with state collection ##
        collector = UnitModelStateCollector()
        model = UnitModel(
            games=self.games,
            config=self.config,
            state_collector=collector
        )
        model.run()
        
        ## compute only missing weeks ##
        missing_set = set(missing_weeks)
        new_records: List[RecalibrationRecord] = []
        
        for state in collector.get_all_states():
            if (state.season, state.for_week) not in missing_set:
                continue
            
            if verbose:
                print(f"Computing recalibration for {state.season} week {state.for_week}...")
            
            recalibrator = UnitRecalibrator(
                games=self.games,
                config=self.config,
                league_baseline=state.league_baseline,
                league_qb=state.league_qb,
                season=state.season,
                week=state.for_week,
                teams=state.teams
            )
            
            if recalibrator.model is not None:
                for team_abbr in recalibrator.model.teams.keys():
                    for unit_type in ['pass_off', 'pass_def', 'rush_off', 'rush_def', 'st_off', 'st_def']:
                        ideal_value = recalibrator.get_ideal_value(team_abbr, unit_type)
                        if ideal_value is not None:
                            new_records.append(RecalibrationRecord(
                                season=state.season,
                                for_week=state.for_week,
                                team=team_abbr,
                                unit_type=unit_type,
                                ideal_value=ideal_value
                            ))
        
        ## add to existing set ##
        self.recal_set.upsert(new_records)
        if verbose:
            print(f"Added {len(new_records)} new recalibration records")
    
    def save(self) -> None:
        '''Persist RecalibrationSet to CSV'''
        self.recal_set.to_csv(self.filepath)
    
    def get_set(self) -> RecalibrationSet:
        '''Return the RecalibrationSet'''
        return self.recal_set

