""" functions for generating patch indices """

import itertools
import numbers

def patch_index_tuples(patch_shape, arr_shape, starts=0, steps=1):
    """ yields index tuples to slice an array

    Parameters
    ----------

    patch_shape : tuple
        shape of the patches

    arr_shape : tuple
        same dimension as patch_shape

    starts : integer or tuple
        starting indices of patches

    steps : integer or tuple
        step width between patches

    """
    ndim = len(arr_shape)

    if not len(patch_shape) == ndim:
        raise ValueError("`patch_shape` is incompatible with `arr_shape`")

    if isinstance(steps, numbers.Number):
        steps = (steps, ) * ndim
    if not len(steps) == ndim:
        raise ValueError("`steps` is incompatible with `arr_shape`")

    if isinstance(starts, numbers.Number):
        starts = (starts, ) * ndim
    if not len(starts) == ndim:
        raise ValueError("`starts` is incompatible with `arr_shape`")

    # make sure that always one patch fits into a patch starting at patch_starts
    patch_starts = mdim_range(starts, tuple(a-x+1 for a, x in zip(arr_shape, patch_shape)), steps)

    init_patch_stop = tuple(s+b for s, b in zip(starts, patch_shape))
    final_patch_stop = tuple(s+1 for s in arr_shape)
    patch_stops = list(mdim_range(init_patch_stop, final_patch_stop, steps))

    # iterate over patches in an array
    for b_sta, b_sto in zip(patch_starts, patch_stops):
        # iterate over dimensions in a patch
        yield tuple(slice(bx, by) for bx, by in zip(b_sta, b_sto))


def mdim_range(starts, stops=None, steps=1):
    """
    multidimensional range.
    mdim_range(stops)
    mdim_range(starts, stops[, steps])

    Parameters
    ----------
    shape : tuple
        shape of the tensor
    step : integer or tuple of the same length as shape
        Indicates step size of the iterator.
        If integer is given, then the step is uniform in all dimensions.
    """

    if stops is None:
        starts, stops = 0, starts

    ndim = len(stops)

    if isinstance(steps, numbers.Number):
        steps = (steps, ) * ndim
    if not len(steps) == ndim:
        raise ValueError("`steps` is incompatible with `stops`")

    if isinstance(starts, numbers.Number):
        starts = (starts, ) * ndim
    if not len(starts) == ndim:
        raise ValueError("`starts` is incompatible with `stops`")

    return itertools.product(*(range(*sss) for sss in zip(starts, stops, steps)))
