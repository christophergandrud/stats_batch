"""
Test the whole batch_mean, batch_var workflow to t-Test
"""

from math import sqrt
import stats_batch as sb
import numpy as np
import numpy.testing as npt
from scipy.stats import ttest_ind_from_stats
from scipy.stats import ttest_ind
from math import sqrt

def batch_mean_var_t_test():
    n = 10_000
    a = np.random.normal(size=n)
    b = np.random.normal(size=n)

    # First batch
    # a -----------
    batch_1_a = a[:100]  
    a_current = sb.mean_var_batch(batch_1_a)
    # b -----------
    batch_1_b = b[:100]  
    b_current = sb.mean_var_batch(batch_1_b)

    # Second batch
    # a -----------
    batch_2_a = a[100:n]
    a_current = sb.mean_var_batch(batch_2_a, a_current[0], 
                                  a_current[2], a_current[3])
    npt.assert_approx_equal(a_current[0], np.mean(a))
    npt.assert_approx_equal(a_current[1], np.var(a), significant=4)

    # b -----------
    batch_2_b = b[100:n]
    b_current = sb.mean_var_batch(batch_2_b, b_current[0], 
                                b_current[2], b_current[3])

    # t-test
    batch_t = ttest_ind_from_stats(mean1=a_current[0], mean2=b_current[0],
                                   std1=sqrt(a_current[1]), std2=sqrt(b_current[1]),
                                   nobs1=a_current[3], nobs2=b_current[3])
    list_t = ttest_ind(a, b)
    assert batch_t[0] == list_t[0]
    assert batch_t[1] == list_t[1]


 