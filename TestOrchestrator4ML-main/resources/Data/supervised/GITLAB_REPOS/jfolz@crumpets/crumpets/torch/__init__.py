from torch import device

__version__ = '3.0.0a5'
public, _, local = __version__.partition('+')
__version_info__ = tuple(public.split('.'))
__version_info__ = tuple(__version_info__[:2]) + tuple(__version_info__[2])
if local:
    __version_info__ += (local,)
__pre_release__ = len(__version_info__) > 3
del public, local


def is_single_torch_device(val):
    """checks if val is a value determining a single cuda device"""
    try:
        device(val)
        return True
    except TypeError:
        return False


def is_cpu_only(val):
    """checks if val is a value determining cpu-only cuda devices"""
    try:
        try:
            res = device(val).type == 'cpu'
        except TypeError:
            res = all([device(d).type == 'cpu' for d in val])
        return res
    except TypeError:
        raise TypeError('Given value {} is neither iterable nor a torch device!'.format(val))
