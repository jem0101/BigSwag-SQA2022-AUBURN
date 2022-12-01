from sympy import Set, symbols, exp, log, S, Wild, Dummy, oo
from sympy.core import Expr, Add
from sympy.core.function import Lambda, _coeff_isneg, FunctionClass
from sympy.logic.boolalg import true
from sympy.multipledispatch import dispatch
from sympy.sets import (imageset, Interval, FiniteSet, Union, ImageSet,
                        EmptySet, Intersection, Range)
from sympy.sets.fancysets import Integers, Naturals, Reals
from sympy.functions.elementary.exponential import match_real_imag


_x, _y = symbols("x y")

FunctionUnion = (FunctionClass, Lambda)


@dispatch(FunctionClass, Set)  # type: ignore # noqa:F811
def _set_function(f, x): # noqa:F811
    return None

@dispatch(FunctionUnion, FiniteSet)  # type: ignore # noqa:F811
def _set_function(f, x): # noqa:F811
    return FiniteSet(*map(f, x))

@dispatch(Lambda, Interval)  # type: ignore # noqa:F811
def _set_function(f, x): # noqa:F811
    from sympy.functions.elementary.miscellaneous import Min, Max
    from sympy.solvers.solveset import solveset
    from sympy.core.function import diff, Lambda
    from sympy.series import limit
    from sympy.calculus.singularities import singularities
    from sympy.sets import Complement
    # TODO: handle functions with infinitely many solutions (eg, sin, tan)
    # TODO: handle multivariate functions

    expr = f.expr
    if len(expr.free_symbols) > 1 or len(f.variables) != 1:
        return
    var = f.variables[0]
    if not var.is_real:
        if expr.subs(var, Dummy(real=True)).is_real is False:
            return

    if expr.is_Piecewise:
        result = S.EmptySet
        domain_set = x
        for (p_expr, p_cond) in expr.args:
            if p_cond is true:
                intrvl = domain_set
            else:
                intrvl = p_cond.as_set()
                intrvl = Intersection(domain_set, intrvl)

            if p_expr.is_Number:
                image = FiniteSet(p_expr)
            else:
                image = imageset(Lambda(var, p_expr), intrvl)
            result = Union(result, image)

            # remove the part which has been `imaged`
            domain_set = Complement(domain_set, intrvl)
            if domain_set is S.EmptySet:
                break
        return result

    if not x.start.is_comparable or not x.end.is_comparable:
        return

    try:
        from sympy.polys.polyutils import _nsort
        sing = list(singularities(expr, var, x))
        if len(sing) > 1:
            sing = _nsort(sing)
    except NotImplementedError:
        return

    if x.left_open:
        _start = limit(expr, var, x.start, dir="+")
    elif x.start not in sing:
        _start = f(x.start)
    if x.right_open:
        _end = limit(expr, var, x.end, dir="-")
    elif x.end not in sing:
        _end = f(x.end)

    if len(sing) == 0:
        soln_expr = solveset(diff(expr, var), var)
        if not (isinstance(soln_expr, FiniteSet) or soln_expr is EmptySet):
            return
        solns = list(soln_expr)

        extr = [_start, _end] + [f(i) for i in solns
                                 if i.is_real and i in x]
        start, end = Min(*extr), Max(*extr)

        left_open, right_open = False, False
        if _start <= _end:
            # the minimum or maximum value can occur simultaneously
            # on both the edge of the interval and in some interior
            # point
            if start == _start and start not in solns:
                left_open = x.left_open
            if end == _end and end not in solns:
                right_open = x.right_open
        else:
            if start == _end and start not in solns:
                left_open = x.right_open
            if end == _start and end not in solns:
                right_open = x.left_open

        return Interval(start, end, left_open, right_open)
    else:
        return imageset(f, Interval(x.start, sing[0],
                                    x.left_open, True)) + \
            Union(*[imageset(f, Interval(sing[i], sing[i + 1], True, True))
                    for i in range(0, len(sing) - 1)]) + \
            imageset(f, Interval(sing[-1], x.end, True, x.right_open))

@dispatch(FunctionClass, Interval)  # type: ignore # noqa:F811
def _set_function(f, x): # noqa:F811
    if f == exp:
        return Interval(exp(x.start), exp(x.end), x.left_open, x.right_open)
    elif f == log:
        return Interval(log(x.start), log(x.end), x.left_open, x.right_open)
    return ImageSet(Lambda(_x, f(_x)), x)

@dispatch(FunctionUnion, Union)  # type: ignore # noqa:F811
def _set_function(f, x): # noqa:F811
    return Union(*(imageset(f, arg) for arg in x.args))

@dispatch(FunctionUnion, Intersection)  # type: ignore # noqa:F811
def _set_function(f, x): # noqa:F811
    from sympy.sets.sets import is_function_invertible_in_set
    # If the function is invertible, intersect the maps of the sets.
    if is_function_invertible_in_set(f, x):
        return Intersection(*(imageset(f, arg) for arg in x.args))
    else:
        return ImageSet(Lambda(_x, f(_x)), x)

@dispatch(FunctionUnion, type(EmptySet))  # type: ignore # noqa:F811
def _set_function(f, x): # noqa:F811
    return x

@dispatch(FunctionUnion, Set)  # type: ignore # noqa:F811
def _set_function(f, x): # noqa:F811
    return ImageSet(Lambda(_x, f(_x)), x)

@dispatch(FunctionUnion, Range)  # type: ignore # noqa:F811
def _set_function(f, self): # noqa:F811
    from sympy.core.function import expand_mul
    if not self:
        return S.EmptySet
    if not isinstance(f.expr, Expr):
        return
    if self.size == 1:
        return FiniteSet(f(self[0]))
    if f is S.IdentityFunction:
        return self

    x = f.variables[0]
    expr = f.expr
    # handle f that is linear in f's variable
    if x not in expr.free_symbols or x in expr.diff(x).free_symbols:
        return
    if self.start.is_finite:
        F = f(self.step*x + self.start)  # for i in range(len(self))
    else:
        F = f(-self.step*x + self[-1])
    F = expand_mul(F)
    if F != expr:
        return imageset(x, F, Range(self.size))

@dispatch(FunctionUnion, Integers)  # type: ignore # noqa:F811
def _set_function(f, self): # noqa:F811
    expr = f.expr
    if not isinstance(expr, Expr):
        return

    n = f.variables[0]
    if expr == abs(n):
        return S.Naturals0

    # f(x) + c and f(-x) + c cover the same integers
    # so choose the form that has the fewest negatives
    c = f(0)
    fx = f(n) - c
    f_x = f(-n) - c
    neg_count = lambda e: sum(_coeff_isneg(_) for _ in Add.make_args(e))
    if neg_count(f_x) < neg_count(fx):
        expr = f_x + c

    a = Wild('a', exclude=[n])
    b = Wild('b', exclude=[n])
    match = expr.match(a*n + b)
    if match and match[a]:
        # canonical shift
        a, b = match[a], match[b]
        if a in [1, -1]:
            # drop integer addends in b
            nonint = []
            for bi in Add.make_args(b):
                if not bi.is_integer:
                    nonint.append(bi)
            b = Add(*nonint)
        if b.is_number and a.is_real:
            # avoid Mod for complex numbers, #11391
            br, bi = match_real_imag(b)
            if br and br.is_comparable and a.is_comparable:
                br %= a
                b = br + S.ImaginaryUnit*bi
        elif b.is_number and a.is_imaginary:
            br, bi = match_real_imag(b)
            ai = a/S.ImaginaryUnit
            if bi and bi.is_comparable and ai.is_comparable:
                bi %= ai
                b = br + S.ImaginaryUnit*bi
        expr = a*n + b

    if expr != f.expr:
        return ImageSet(Lambda(n, expr), S.Integers)


@dispatch(FunctionUnion, Naturals)  # type: ignore # noqa:F811
def _set_function(f, self): # noqa:F811
    expr = f.expr
    if not isinstance(expr, Expr):
        return

    x = f.variables[0]
    if not expr.free_symbols - {x}:
        if expr == abs(x):
            if self is S.Naturals:
                return self
            return S.Naturals0
        step = expr.coeff(x)
        c = expr.subs(x, 0)
        if c.is_Integer and step.is_Integer and expr == step*x + c:
            if self is S.Naturals:
                c += step
            if step > 0:
                if step == 1:
                    if c == 0:
                        return S.Naturals0
                    elif c == 1:
                        return S.Naturals
                return Range(c, oo, step)
            return Range(c, -oo, step)


@dispatch(FunctionUnion, Reals)  # type: ignore # noqa:F811
def _set_function(f, self): # noqa:F811
    expr = f.expr
    if not isinstance(expr, Expr):
        return
    return _set_function(f, Interval(-oo, oo))
