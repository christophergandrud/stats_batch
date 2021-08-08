"""
Utilities for testing and illustrating stats-batch
"""

from itertools import islice

def group_elements(lst, batch_size):
    """
    Group elements of a list into chunks of size `batch_size`. 
    This is a utility to illustrate stats-batch functions.

    Parameters
    ----------
    lst : list
        The list to be grouped into batches
    chunk_size : int
        The size of the batches
    """
    lst = iter(lst)
    return iter(lambda: tuple(islice(lst, batch_size)), ())