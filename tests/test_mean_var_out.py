import stats_batch as sb
import pandas as pd
import numpy as np

def test_mean_var_out():
    """
    Test if sb.mean_var_batch correctly outputs to pandas and CSV
    """
    sb_pd = sb.mean_var_batch([1,2,3,4]).to_pandas()
    assert isinstance(sb_pd, pd.DataFrame)

def test_batch_updating_to_pandas():
    """
    Test workflow that updates pandas data frame with each new batch
    """
    # Set up example ----------------------------------------------
    # Simulate full sample
    n = 10_000
    batch_size = 1_000
    a = np.random.normal(size=n, loc=0.1)

    for i, new_list in enumerate(sb.group_elements(a , batch_size)):
        if i == 0:
            mean_var_a = sb.mean_var_batch(new_list)
            suf_stats_df = mean_var_a.to_pandas()
        else:
            mean_var_a.update(new_list)
            suf_stats_df = suf_stats_df.append(mean_var_a.to_pandas())

    assert isinstance(suf_stats_df, pd.DataFrame)
    assert len(suf_stats_df) == n/batch_size

