from cupy import elementwise


def _create_float_test_ufunc(name, doc):
    return elementwise.create_ufunc(
        'cupy_' + name, ['e->?', 'f->?', 'd->?'], 'out0 = %s(in0)' % name,
        doc=doc)


isfinite = _create_float_test_ufunc(
    'isfinite',
    '''Tests finiteness elementwise.

    Each element of returned array is True only if the corresponding element
    of the input is finite (i.e. not an infinity nor NaN).

    .. seealso:: :data:`numpy.isfinite`

    ''')


isinf = _create_float_test_ufunc(
    'isinf',
    '''Tests if each element is the positive or negative infinity.

    .. seealso:: :data:`numpy.isinf`

    ''')


isnan = _create_float_test_ufunc(
    'isnan',
    '''Tests if each element is a NaN.

    .. seealso:: :data:`numpy.isnan`

    ''')


def isneginf(x, y=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def isposinf(x, y=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError
