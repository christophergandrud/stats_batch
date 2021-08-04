"""
Functions to use batch algorithms to find the mean and variance of a sample.
"""

import numpy as np
from scipy.stats import ttest_ind_from_stats
import pandas as pd
from math import sqrt

def sum_square_deviations(x) -> float:
    """
    Sum of the squared deviations of a sample from the mean.

    Parameters
    ----------
    x : array_like
        Sample to find the sum of the squared deviations of.

    Returns
    -------
    float 
        sum of the squared deviations of the sample from the mean.

    Examples
    --------
    >>> from stats_batch import sum_square_deviations
    >>> sum_square_deviations([1, 2, 3])
    2.0
    """
    return sum((x - np.mean(x)) ** 2)


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
        tuple(float, int)
        1. The mean from the new batch and the mean of the prior batches. This will be *approximately* 
        equal to the total sample mean.
        If only `new_batch` is given, the mean of the batch is returned with the total sample size without updating.
        2. Total sample size. 

    Examples
    -------- 
    >>> from stats_batch import mean_batch
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


def var_batch(new_batch, prior_mean:float=None, prior_sum_squares:float=None, prior_sample_size:int=None):
    """
    Find the new (approximate) variance of a sample updated by one batch.
    If only `new_batch` is supplied, `np.var` is used.

    Parameters
    ----------
        new_batch: List[Union[int, float]]
            List of all values in the new batch
        prior_mean: float
            Mean up until the new batch.
        prior_sum_squares: float
            Sum of the squares of the prior batch.
        prior_sample_size: int
            Number of samples in the prior batches.
    
    Returns
    -------
        tuple(float, float)
            1. The variance of the new batch and prior batches.
            2. The sum of square deviations of the new batch and prior batches.

    Examples
    --------
    >>> from stats_batch import var_batch
    >>> var_batch([1,2,3,4])
    (1.25, 5.0)
    >>> var_batch([1,2,3,4], prior_mean = 0, prior_sum_squares = 5, prior_sample_size = 4)
    (3.2142857142857144, 22.5)

    References
    ----------
    See Gandrud (2021) <https://elegant-heyrovsky-54a43f.netlify.app/privacy-first-ds-mean-var.html> for algorithm details.

    This work is based on:
    - Chou (2021, 5) <https://arxiv.org/pdf/2102.03316.pdf>

    - Chan et al (1983) <http://www.cs.yale.edu/publications/techreports/tr222.pdf> 
    """
    if prior_sum_squares is None or prior_mean is None or prior_sample_size is None:
        return (np.var(new_batch), sum_square_deviations(new_batch))
    else:
        batch_mean = np.mean(new_batch)
        total_samples = prior_sample_size + len(new_batch)
        ssd_new_batch = sum_square_deviations(new_batch)
        new_ssd = (prior_sum_squares + ssd_new_batch) + \
            (prior_sample_size / total_samples) * \
            (total_samples - prior_sample_size) * \
            (batch_mean - prior_mean) ** 2
        var_new = new_ssd / (total_samples - 1)
        return (var_new, new_ssd)

class MeanVarBatch:
    """
    Class for mean and variance of a sample created through batch updating.
    """
    def __init__(self, mean, var, sum_squares, sample_size):
        self.mean = mean
        self.var = var
        self.sum_squares = sum_squares
        self.sample_size = sample_size

    def __getitem__(self, item):
        return getattr(self, item)

    def to_pandas(self):
        """
        Returns a pandas dataframe with the mean, variance, sum of squared deviations 
        and sample size.

        Returns
        -------
        pandas.DataFrame

        Examples
        --------
        >>> sb.mean_var_batch([1,2,3,4]).to_pandas()
           mean   var  sum_squared_dev  sample_size
        0   2.5  1.25              5.0            4
        """
        return pd.DataFrame({"mean": [self.mean], "var": [self.var], 
        "sum_squared_dev": [self.sum_squares], "sample_size": [self.sample_size]})

    def print(self):
        return self.to_pandas()

    def update(self, new_batch):
        """
        Updates the existing object with a new batch.

        Parameters
        ----------
        new_batch: List[Union[int, float]]
            List of all values in the new batch
        """
        updated = mean_var_batch(new_batch, self.mean, self.sum_squares, self.sample_size)
        self.__dict__.update(updated.__dict__)

    def ttest_ind(self, other):
        """
        T-test for means of two independent samples

        Parameters
        ----------
        other : MeanVarBatch
            The other sample's sufficient statistics as generated by `mean_var_batch` 

        Returns
        -------
        (float, float)
            The t-statistic and P-value.
        """
        return ttest_ind_from_stats(mean1=self.mean, mean2=other.mean,
                                   std1=sqrt(self.var), std2=sqrt(other.var),
                                   nobs1=self.sample_size, nobs2=other.sample_size)    

    def to_csv(self, filename):
        """
        Saves mean, variance, sum of squared deviations, and sample size to a csv file.

        Parameters
        ----------
        filename : str
            The name of the file to save.

        Examples
        --------
        >>> sb.mean_var_batch([1,2,3,4]).to_csv("test.csv")
        """
        self.to_pandas().to_csv(filename)

def mean_var_batch(new_batch, prior_mean:float=None, prior_sum_squares:float=None, prior_sample_size:int=None):
    """
    Find the new (approximate) mean and variance of a sample updated by one batch.
    If only `new_batch` is supplied, `np.mean` and `np.var` are used.

    Parameters
    ----------
        new_batch: List[Union[int, float]]
            List of all values in the new batch
        prior_mean: float
            Mean up until the new batch.
        prior_sum_squares: float
            Sum of the squares of the prior batch.
        prior_sample_size: int
            Number of samples in the prior batches.
    
    Returns
    -------
        tuple(float, float, float, int) 
        1. The mean of the new batch and prior batches.
        2. The variance of the new batch and prior batches.
        3. The sum of square deviations of the new batch and prior batches.
        4. The sample size of the new batch and prior batches.

    Examples
    --------
    >>> from stats_batch import mean_var_batch
    >>> mean_var_batch([1,2,3,4])
    >>> mean_var_batch([1,2,3,4], prior_mean = 2.5, prior_sum_squares = 5, prior_sample_size = 4)
    """
    b_mean, b_n = mean_batch(new_batch, prior_mean, prior_sample_size)
    b_var, b_ssd = var_batch(new_batch, prior_mean, prior_sum_squares, prior_sample_size)
    return  MeanVarBatch(b_mean, b_var, b_ssd, b_n)

        