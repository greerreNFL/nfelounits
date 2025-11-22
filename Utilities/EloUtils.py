'''
EloUtils Module

Utility functions for elo-based calculations.
'''


def calculate_win_probability(elo_diff: float) -> float:
    '''
    Calculate win probability from elo difference
    
    Uses standard elo formula: 1 / (1 + 10^(-elo_diff/400))
    
    Parameters:
    * elo_diff: Elo difference (home team elo - away team elo)
    
    Returns:
    * Win probability for the team with higher elo (0 to 1)
    '''
    return 1.0 / (1.0 + 10 ** (-elo_diff / 400))

