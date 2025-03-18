"""
Evaluation metrics for recommendation systems.
"""

import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error.
    
    Args:
        y_true (np.ndarray): Array of true ratings
        y_pred (np.ndarray): Array of predicted ratings
        
    Returns:
        float: RMSE score
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true (np.ndarray): Array of true ratings
        y_pred (np.ndarray): Array of predicted ratings
        
    Returns:
        float: MAE score
    """
    return mean_absolute_error(y_true, y_pred)

def precision_at_k(actual: List[int], predicted: List[int], k: int) -> float:
    """
    Calculate precision@k for recommendation systems.
    
    Args:
        actual (List[int]): List of actual relevant items
        predicted (List[int]): List of predicted items
        k (int): Number of recommendations to consider
        
    Returns:
        float: Precision@k score
    """
    if len(predicted) > k:
        predicted = predicted[:k]
    
    score = len(set(actual) & set(predicted)) / float(min(k, len(predicted)))
    return score

def recall_at_k(actual: List[int], predicted: List[int], k: int) -> float:
    """
    Calculate recall@k for recommendation systems.
    
    Args:
        actual (List[int]): List of actual relevant items
        predicted (List[int]): List of predicted items
        k (int): Number of recommendations to consider
        
    Returns:
        float: Recall@k score
    """
    if len(predicted) > k:
        predicted = predicted[:k]
    
    score = len(set(actual) & set(predicted)) / float(len(actual))
    return score

def ndcg_at_k(actual: List[int], predicted: List[int], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) at k.
    
    Args:
        actual (List[int]): List of actual relevant items
        predicted (List[int]): List of predicted items
        k (int): Number of recommendations to consider
        
    Returns:
        float: NDCG@k score
    """
    if len(predicted) > k:
        predicted = predicted[:k]
    
    dcg = 0.0
    idcg = 0.0
    
    for i, item in enumerate(predicted):
        if item in actual:
            # Calculate DCG
            dcg += 1.0 / np.log2(i + 2)  # i + 2 because i starts at 0
    
    # Calculate IDCG
    for i in range(min(len(actual), k)):
        idcg += 1.0 / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_recommendations(
    recommender,
    test_data: Dict[int, List[int]],
    k: int = 10
) -> Dict[str, float]:
    """
    Evaluate a recommender system using multiple metrics.
    
    Args:
        recommender: Recommender system instance
        test_data (Dict[int, List[int]]): Dictionary of user_id to list of actual items
        k (int): Number of recommendations to consider
        
    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics
    """
    try:
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        
        for user_id, actual_items in test_data.items():
            # Get recommendations for user
            recommendations = recommender.predict(user_id, k)
            predicted_items = [item_id for item_id, _ in recommendations]
            
            # Calculate metrics
            precision_scores.append(precision_at_k(actual_items, predicted_items, k))
            recall_scores.append(recall_at_k(actual_items, predicted_items, k))
            ndcg_scores.append(ndcg_at_k(actual_items, predicted_items, k))
        
        # Calculate average metrics
        metrics = {
            f'precision@{k}': np.mean(precision_scores),
            f'recall@{k}': np.mean(recall_scores),
            f'ndcg@{k}': np.mean(ndcg_scores)
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise 