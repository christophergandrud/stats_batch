"""
Functions to use batch algorithms to find the mean and variance of a sample.
"""

import numpy as np


def mean_batch(new_batch, prior_sample_size:int=None, prior_mean:float=None) -> float:
    """
    Find the new mean of a sample updated by one batch. If only `new_batch` is 
    given, then `np.mean` is used.

    Parameters
    ----------
        new_batch: List[Union[int, float]]  
            List of all values in the new batch
        prior_sample_size: int
            Number of samples in the prior batches.
        prior_mean: float
            Mean up until the new batch.

    Returns
    -------
        The mean including the new batch and mean of the prior batches or, if only `new_batch` is given, the
        mean of the batch without updating.

    Examples
    -------- 
    >>> mean_batch([1,2,3,4])
    2.5
    >>> mean_batch([1,2,3,4], 4, 0)
    1.25

    Sources
    -------
    See Gandrud (2021) <https://elegant-heyrovsky-54a43f.netlify.app/privacy-first-ds-mean-var.html> for algorithm details. 
    """
    if prior_sample_size is None or prior_mean is None:
        return np.mean(new_batch)
    else:
        sum_new_batch = sum(new_batch)
        total_samples = prior_sample_size + len(new_batch)
        return prior_mean + (1/total_samples) * (sum_new_batch - ((total_samples - prior_sample_size) * prior_mean))


