import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score
import json
from tqdm import tqdm

def load_yelp_data(file_path, n_samples=25000):
    """Load Yelp dataset with specified number of samples"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f)):
            if i >= n_samples:
                break
            data.append(json.loads(line))
    return pd.DataFrame(data)

def save_metrics_to_csv(metrics_dict, filename='model_metrics.csv'):
    """Save model metrics to CSV file"""
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv(filename, index=False)
    return metrics_df

def calculate_metrics(y_true, y_pred):
    """Calculate all required metrics"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    } 