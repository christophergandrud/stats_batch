"""
Functions to use batch algorithms to find the mean and variance of a sample.
"""

import numpy as np

def mean_batch(new_batch, prior_mean:float=None, prior_sample_size:int=None):
    """
    Find the new (approximate) mean of a sample updated by one batch. 
    If only `new_batch` is given, then `np.mean` is used.

    Parameters
    ----------
        new_batch: List[Union[int, float]]  
            List of all values in the new batch
        prior_mean: float
            Mean up until the new batch.
        prior_sample_size: int
            Number of samples in the prior batches.

    Returns
    -------
        1. The mean from the new batch and the mean of the prior batches. This will be *approximately* 
        equal to the total sample mean.
        If only `new_batch` is given, the mean of the batch is returned with the total sample size without updating.
        2. Total sample size. 

    Examples
    -------- 
    >>> mean_batch([1,2,3,4])
    (2.5, 4)
    >>> mean_batch([1,2,3,4], prior_mean = 0, prior_sample_size = 4, )
    (1.25, 8)

    References
    ----------
    See Gandrud (2021) <https://elegant-heyrovsky-54a43f.netlify.app/privacy-first-ds-mean-var.html> for algorithm details.

    This work is based on:
    - Chou (2021, 5) <https://arxiv.org/pdf/2102.03316.pdf>

    - Chan et al (1983) <http://www.cs.yale.edu/publications/techreports/tr222.pdf> 
    """
    if prior_sample_size is None or prior_mean is None:
        return np.mean(new_batch), len(new_batch)
    else:
        sum_new_batch = sum(new_batch)
        total_samples = prior_sample_size + len(new_batch)
        updated_mean = prior_mean + (1/total_samples) * (sum_new_batch - ((total_samples - prior_sample_size) * prior_mean))
        return (updated_mean, total_samples)

