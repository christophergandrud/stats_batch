import stats_batch as sb
import numpy as np
import numpy.testing as npt

# Test that var_batch returns exact `np.var` variance if there is 
# no prior batch information
def test_var_batch_missing_prior():
    x = list(range(1, 100))
    assert sb.var_batch(x)[0] == np.var(x)
    assert sb.var_batch(x)[1] == sb.sum_square_deviations(x)  

# Test that var_batch returns approximate `np.var` for batch updates
def test_var_batch_multiple_batches():
    n = 10_000
    x = np.random.normal(size=n)

    # First batch
    batch_1 = x[:100]  
    b1_mean, b1_n = sb.mean_batch(batch_1)
    b1_var, b1_ssd = sb.var_batch(batch_1) 

    # Second batch
    batch_2 = x[100:n]
    b2_var, b2_ssd = sb.var_batch(batch_2, b1_mean, b1_ssd, b1_n)
    npt.assert_approx_equal(b2_var, np.var(x), significant=4)
    npt.assert_approx_equal(b2_ssd, sb.sum_square_deviations(x))
