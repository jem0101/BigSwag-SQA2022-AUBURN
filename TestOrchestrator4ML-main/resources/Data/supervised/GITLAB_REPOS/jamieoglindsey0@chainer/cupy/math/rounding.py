from cupy.math import ufunc


def around(a, decimals=0, out=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


# TODO(beam2d): Implement it
# round_ = around


rint = ufunc.create_math_ufunc(
    'rint', 1, 'cupy_rint',
    '''Rounds eacy element of an array to the nearest integer.

    .. seealso:: :data:`numpy.rint`

    ''')


def fix(x, y=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


floor = ufunc.create_math_ufunc(
    'floor', 1, 'cupy_floor',
    '''Rounds each element of an array to its floow integer.

    .. seealso:: :data:`numpy.floor`

    ''')


ceil = ufunc.create_math_ufunc(
    'ceil', 1, 'cupy_ceil',
    '''Rounds each element of an array to its ceil integer.

    .. seealso:: :data:`numpy.ceil`

    ''')


trunc = ufunc.create_math_ufunc(
    'trunc', 1, 'cupy_trunc',
    '''Rounds each element of an array towards zero.

    .. seealso:: :data:`numpy.trunc`

    ''')
