import numpy as np

'''
Helpers for statistics. Mainly for doing statistical inference using z-scores to determin and
get rid of outliers :)

Author: Jason Ho
'''

def normalize(vector):
    global_min = np.min(vector)
    global_max = np.max(vector)

    if global_min == global_max:
        return vector

    normalized = (vector - global_min) / (global_max - global_min)
    return normalized

def min_max_normalization(vector):
    """
    Performs min-max normalization on the input vector, scaling its values to the range [0, 1].

    Parameters:
    - vector: NumPy array or list containing the values to be normalized.

    Returns:
    - normalized: The normalized vector.
    - global_min: The minimum value of the original vector.
    - global_max: The maximum value of the original vector.
    """
    global_min = np.min(vector)
    global_max = np.max(vector)

    if global_max - global_min != 0:
        normalized = (vector - global_min) / (global_max - global_min)
    else:
        normalized = vector

    return normalized, global_min, global_max

def reverse_min_max_normalization(vector, global_min, global_max):
    """
    Reverses the min-max normalization process on the input vector using provided min and max values.

    Parameters:
    - vector: NumPy array or list containing the normalized values.
    - global_min: The minimum value of the original vector.
    - global_max: The maximum value of the original vector.

    Returns:
    - reversed_vector: The vector after reverse min-max normalization.
    """
    return (vector * (global_max - global_min)) + global_min

def calculate_z_score(vector):
    """
    Calculates the z-scores for the values in the input vector.

    Parameters:
    - vector: NumPy array or list containing the values.

    Returns:
    - z_scores: The z-scores of the input vector.
    """
    mean_value = np.mean(vector)
    std_dev = np.std(vector)
    z_scores = (vector - mean_value) / std_dev
    return z_scores

def generate_outlier_vector(z_scores, threshold=3.0):
    """
    Generates a binary vector indicating outliers based on z-scores and a specified threshold.

    Parameters:
    - z_scores: NumPy array or list containing z-scores.
    - threshold: The threshold value for determining outliers (default is 3.0).

    Returns:
    - outlier_vector: Binary vector indicating outliers (1 for outliers, 0 otherwise).
    """
    outliers = z_scores > threshold
    outlier_vector = np.zeros_like(z_scores)
    outlier_vector[outliers] = 1
    return outlier_vector
