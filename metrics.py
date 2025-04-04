import numpy as np
from scipy.spatial.distance import cdist

def dice(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

def iou(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / union


def specificity(y_true, y_pred):
    true_negative = np.sum((1 - y_true) * (1 - y_pred))
    false_positive = np.sum((1 - y_true) * y_pred)
    return true_negative / (true_negative + false_positive)


# sensitivity
def recall(y_true, y_pred):
    true_positive = np.sum(y_true * y_pred)
    false_negative = np.sum(y_true * (1 - y_pred))
    return true_positive / (true_positive + false_negative)


def hausdorff(y_true, y_pred):
    A = np.argwhere(y_true) 
    B = np.argwhere(y_pred)  
    
    if len(A) == 0 or len(B) == 0:
        if len(A) == 0 and len(B) == 0:
            return 0.0  # Both empty, distance is 0
        return float('inf')  # One is empty, distance is infinite
    
    distances = cdist(A, B)
    
    d_AB = np.max(np.min(distances, axis=1))
    d_BA = np.max(np.min(distances, axis=0))
    
    return max(d_AB, d_BA)

