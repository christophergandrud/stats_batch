import stats_batch as sb
import numpy as np

# Test mean_batch returns the mean if prior_mean and prior_sample_size are missing
def test_mean_batch_missing_prior_mean_prior_sample_size():
    x = list(range(1, 100))
    assert sb.mean_batch(x) == np.mean(x) 