"""
Utilities for testing and illustrating stats-batch
"""

from itertools import islice

def group_elements(lst:list, batch_size:int) -> list:
    """
    Group elements of a list into chunks of size `bpipatch_size`. 
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