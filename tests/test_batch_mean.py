import stats_batch as sb
import numpy as np
import numpy.testing as npt

# Test mean_batch returns the mean if prior_mean and prior_sample_size are missing
def test_mean_batch_missing_prior_mean_prior_sample_size():
    x = list(range(1, 100))
    assert sb.mean_batch(x)[0] == np.mean(x) 
    assert sb.mean_batch(x)[1] == len(x)

# Test mean_batch returns the correct mean from multiple batches
def test_mean_batch_multiple_batches():
    n = 10_000
    x = np.random.normal(size=n)

    # First batch
    batch_1 = x[:100]  
    b1_mean, b1_n = sb.mean_batch(batch_1) 

    # Second batch
    batch_2 = x[100:n]
    b2_mean, b2_n = sb.mean_batch(batch_2, b1_mean, b1_n)
    npt.assert_approx_equal(b2_mean, np.mean(x))
    assert b2_n == n
