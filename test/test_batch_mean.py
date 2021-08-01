import stats_batch as sb
import numpy as np

# Test batch_mean returns the mean if prior_mean and prior_sample_size are missing
def test_batch_mean_missing_prior_mean_prior_sample_size():
    x = list(range(1, 100))
    assert sb.batch_mean(x) == np.mean(x) 