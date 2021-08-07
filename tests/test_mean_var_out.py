import stats_batch as sb
import pandas as pd

def test_mean_var_out():
    """
    Test if sb.mean_var_batch correctly outputs to pandas and CSV
    """
    sb_pd = sb.mean_var_batch([1,2,3,4]).to_pandas()
    assert isinstance(sb_pd, pd.DataFrame)