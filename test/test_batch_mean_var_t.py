from math import sqrt
import stats_batch as sb
import numpy as np
from pytest import approx
from scipy.stats import ttest_ind_from_stats
from scipy.stats import ttest_ind

def test_batch_mean_var_t_test():
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
    a_current.update(batch_2_a)
    assert a_current.mean == approx(np.mean(a))
    assert a_current.var  == approx(np.var(a), rel = 1e-3)

    # b -----------
    batch_2_b = b[100:n]
    b_current.update(batch_2_b)
    assert b_current.mean == approx(np.mean(b))
    assert b_current.var == approx(np.var(b), rel = 1e-3)

    # t-test
    batch_t = a_current.ttest_ind(b_current)
    list_t = ttest_ind(a, b)
    assert batch_t[0] == approx(list_t[0])
    assert batch_t[1] == approx(list_t[1])


 

