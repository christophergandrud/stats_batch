import stats_batch as sb
import numpy as np
import numpy.testing as npt

# Test that var_batch returns exact `np.var` variance if there is 
# no prior batch information
def test_var_batch_missing_prior():
    x = list(range(1, 100))
    assert sb.var_batch(x) == np.var(x)  