__version__ = '3.0.0a5'
public, _, local = __version__.partition('+')
__version_info__ = tuple(public.split('.'))
__version_info__ = tuple(__version_info__[:2]) + tuple(__version_info__[2])
if local:
    __version_info__ += (local,)
__pre_release__ = len(__version_info__) > 3
del public, local
