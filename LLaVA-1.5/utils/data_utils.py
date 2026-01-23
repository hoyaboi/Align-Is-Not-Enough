"""
Data loading utilities for LLaVA attack
"""
import pandas as pd
from pathlib import Path


def get_goals_and_targets(train_data_path, test_data_path=None, 
                         n_train_data=520, n_test_data=0, offset=0):
    """
    Load goals and targets from CSV files.
    
    Args:
        train_data_path: Path to training data CSV
        test_data_path: Path to test data CSV (optional)
        n_train_data: Number of training samples to load
        n_test_data: Number of test samples to load
        offset: Offset for data loading
    
    Returns:
        train_goals, train_targets, test_goals, test_targets
    """
    train_data = pd.read_csv(train_data_path)
    train_targets = train_data['target'].tolist()[offset:offset+n_train_data]
    
    if 'goal' in train_data.columns:
        train_goals = train_data['goal'].tolist()[offset:offset+n_train_data]
    else:
        train_goals = [""] * len(train_targets)
    
    test_goals = []
    test_targets = []
    
    if test_data_path and Path(test_data_path).exists() and n_test_data > 0:
        test_data = pd.read_csv(test_data_path)
        test_targets = test_data['target'].tolist()[offset:offset+n_test_data]
        if 'goal' in test_data.columns:
            test_goals = test_data['goal'].tolist()[offset:offset+n_test_data]
        else:
            test_goals = [""] * len(test_targets)
    elif n_test_data > 0:
        test_targets = train_data['target'].tolist()[offset+n_train_data:offset+n_train_data+n_test_data]
        if 'goal' in train_data.columns:
            test_goals = train_data['goal'].tolist()[offset+n_train_data:offset+n_train_data+n_test_data]
        else:
            test_goals = [""] * len(test_targets)
    
    assert len(train_goals) == len(train_targets)
    assert len(test_goals) == len(test_targets)
    
    print(f'Loaded {len(train_goals)} train goals')
    print(f'Loaded {len(test_goals)} test goals')
    
    return train_goals, train_targets, test_goals, test_targets
