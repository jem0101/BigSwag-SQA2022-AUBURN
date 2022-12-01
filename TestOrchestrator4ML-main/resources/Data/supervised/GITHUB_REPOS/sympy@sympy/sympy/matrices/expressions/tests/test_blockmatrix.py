from sympy import Trace
from sympy.testing.pytest import raises, slow
from sympy.matrices.expressions.blockmatrix import (
    block_collapse, bc_matmul, bc_block_plus_ident, BlockDiagMatrix,
    BlockMatrix, bc_dist, bc_matadd, bc_transpose, bc_inverse,
    blockcut, reblock_2x2, deblock)
from sympy.matrices.expressions import (MatrixSymbol, Identity,
        Inverse, trace, Transpose, det, ZeroMatrix)
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.matrices import (
    Matrix, ImmutableMatrix, ImmutableSparseMatrix)
from sympy.core import Tuple, symbols, Expr
from sympy.functions import transpose

i, j, k, l, m, n, p = symbols('i:n, p', integer=True)
A = MatrixSymbol('A', n, n)
B = MatrixSymbol('B', n, n)
C = MatrixSymbol('C', n, n)
D = MatrixSymbol('D', n, n)
G = MatrixSymbol('G', n, n)
H = MatrixSymbol('H', n, n)
b1 = BlockMatrix([[G, H]])
b2 = BlockMatrix([[G], [H]])

def test_bc_matmul():
    assert bc_matmul(H*b1*b2*G) == BlockMatrix([[(H*G*G + H*H*H)*G]])

def test_bc_matadd():
    assert bc_matadd(BlockMatrix([[G, H]]) + BlockMatrix([[H, H]])) == \
            BlockMatrix([[G+H, H+H]])

def test_bc_transpose():
    assert bc_transpose(Transpose(BlockMatrix([[A, B], [C, D]]))) == \
            BlockMatrix([[A.T, C.T], [B.T, D.T]])

def test_bc_dist_diag():
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', m, m)
    C = MatrixSymbol('C', l, l)
    X = BlockDiagMatrix(A, B, C)

    assert bc_dist(X+X).equals(BlockDiagMatrix(2*A, 2*B, 2*C))

def test_block_plus_ident():
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', n, m)
    C = MatrixSymbol('C', m, n)
    D = MatrixSymbol('D', m, m)
    X = BlockMatrix([[A, B], [C, D]])
    Z = MatrixSymbol('Z', n + m, n + m)
    assert bc_block_plus_ident(X + Identity(m + n) + Z) == \
            BlockDiagMatrix(Identity(n), Identity(m)) + X + Z

def test_BlockMatrix():
    A = MatrixSymbol('A', n, m)
    B = MatrixSymbol('B', n, k)
    C = MatrixSymbol('C', l, m)
    D = MatrixSymbol('D', l, k)
    M = MatrixSymbol('M', m + k, p)
    N = MatrixSymbol('N', l + n, k + m)
    X = BlockMatrix(Matrix([[A, B], [C, D]]))

    assert X.__class__(*X.args) == X

    # block_collapse does nothing on normal inputs
    E = MatrixSymbol('E', n, m)
    assert block_collapse(A + 2*E) == A + 2*E
    F = MatrixSymbol('F', m, m)
    assert block_collapse(E.T*A*F) == E.T*A*F

    assert X.shape == (l + n, k + m)
    assert X.blockshape == (2, 2)
    assert transpose(X) == BlockMatrix(Matrix([[A.T, C.T], [B.T, D.T]]))
    assert transpose(X).shape == X.shape[::-1]

    # Test that BlockMatrices and MatrixSymbols can still mix
    assert (X*M).is_MatMul
    assert X._blockmul(M).is_MatMul
    assert (X*M).shape == (n + l, p)
    assert (X + N).is_MatAdd
    assert X._blockadd(N).is_MatAdd
    assert (X + N).shape == X.shape

    E = MatrixSymbol('E', m, 1)
    F = MatrixSymbol('F', k, 1)

    Y = BlockMatrix(Matrix([[E], [F]]))

    assert (X*Y).shape == (l + n, 1)
    assert block_collapse(X*Y).blocks[0, 0] == A*E + B*F
    assert block_collapse(X*Y).blocks[1, 0] == C*E + D*F

    # block_collapse passes down into container objects, transposes, and inverse
    assert block_collapse(transpose(X*Y)) == transpose(block_collapse(X*Y))
    assert block_collapse(Tuple(X*Y, 2*X)) == (
        block_collapse(X*Y), block_collapse(2*X))

    # Make sure that MatrixSymbols will enter 1x1 BlockMatrix if it simplifies
    Ab = BlockMatrix([[A]])
    Z = MatrixSymbol('Z', *A.shape)
    assert block_collapse(Ab + Z) == A + Z

def test_block_collapse_explicit_matrices():
    A = Matrix([[1, 2], [3, 4]])
    assert block_collapse(BlockMatrix([[A]])) == A

    A = ImmutableSparseMatrix([[1, 2], [3, 4]])
    assert block_collapse(BlockMatrix([[A]])) == A

def test_issue_17624():
    a = MatrixSymbol("a", 2, 2)
    z = ZeroMatrix(2, 2)
    b = BlockMatrix([[a, z], [z, z]])
    assert block_collapse(b * b) == BlockMatrix([[a**2, z], [z, z]])
    assert block_collapse(b * b * b) == BlockMatrix([[a**3, z], [z, z]])

def test_issue_18618():
    A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert A == Matrix(BlockDiagMatrix(A))

def test_BlockMatrix_trace():
    A, B, C, D = [MatrixSymbol(s, 3, 3) for s in 'ABCD']
    X = BlockMatrix([[A, B], [C, D]])
    assert trace(X) == trace(A) + trace(D)
    assert trace(BlockMatrix([ZeroMatrix(n, n)])) == 0

def test_BlockMatrix_Determinant():
    A, B, C, D = [MatrixSymbol(s, 3, 3) for s in 'ABCD']
    X = BlockMatrix([[A, B], [C, D]])
    from sympy import assuming, Q
    with assuming(Q.invertible(A)):
        assert det(X) == det(A) * det(D - C*A.I*B)

    assert isinstance(det(X), Expr)
    assert det(BlockMatrix([A])) == det(A)
    assert det(BlockMatrix([ZeroMatrix(n, n)])) == 0

def test_squareBlockMatrix():
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', n, m)
    C = MatrixSymbol('C', m, n)
    D = MatrixSymbol('D', m, m)
    X = BlockMatrix([[A, B], [C, D]])
    Y = BlockMatrix([[A]])

    assert X.is_square

    Q = X + Identity(m + n)
    assert (block_collapse(Q) ==
        BlockMatrix([[A + Identity(n), B], [C, D + Identity(m)]]))

    assert (X + MatrixSymbol('Q', n + m, n + m)).is_MatAdd
    assert (X * MatrixSymbol('Q', n + m, n + m)).is_MatMul

    assert block_collapse(Y.I) == A.I

    assert isinstance(X.inverse(), Inverse)

    assert not X.is_Identity

    Z = BlockMatrix([[Identity(n), B], [C, D]])
    assert not Z.is_Identity


def test_BlockMatrix_2x2_inverse_symbolic():
    A = MatrixSymbol('A', n, m)
    B = MatrixSymbol('B', n, k - m)
    C = MatrixSymbol('C', k - n, m)
    D = MatrixSymbol('D', k - n, k - m)
    X = BlockMatrix([[A, B], [C, D]])
    assert X.is_square and X.shape == (k, k)
    assert isinstance(block_collapse(X.I), Inverse)  # Can't invert when none of the blocks is square

    # test code path where only A is invertible
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', n, m)
    C = MatrixSymbol('C', m, n)
    D = ZeroMatrix(m, m)
    X = BlockMatrix([[A, B], [C, D]])
    assert block_collapse(X.inverse()) == BlockMatrix([
        [A.I + A.I * B * (D - C * A.I * B).I * C * A.I, -A.I * B * (D - C * A.I * B).I],
        [-(D - C * A.I * B).I * C * A.I, (D - C * A.I * B).I],
    ])

    # test code path where only B is invertible
    A = MatrixSymbol('A', n, m)
    B = MatrixSymbol('B', n, n)
    C = ZeroMatrix(m, m)
    D = MatrixSymbol('D', m, n)
    X = BlockMatrix([[A, B], [C, D]])
    assert block_collapse(X.inverse()) == BlockMatrix([
        [-(C - D * B.I * A).I * D * B.I, (C - D * B.I * A).I],
        [B.I + B.I * A * (C - D * B.I * A).I * D * B.I, -B.I * A * (C - D * B.I * A).I],
    ])

    # test code path where only C is invertible
    A = MatrixSymbol('A', n, m)
    B = ZeroMatrix(n, n)
    C = MatrixSymbol('C', m, m)
    D = MatrixSymbol('D', m, n)
    X = BlockMatrix([[A, B], [C, D]])
    assert block_collapse(X.inverse()) == BlockMatrix([
        [-C.I * D * (B - A * C.I * D).I, C.I + C.I * D * (B - A * C.I * D).I * A * C.I],
        [(B - A * C.I * D).I, -(B - A * C.I * D).I * A * C.I],
    ])

    # test code path where only D is invertible
    A = ZeroMatrix(n, n)
    B = MatrixSymbol('B', n, m)
    C = MatrixSymbol('C', m, n)
    D = MatrixSymbol('D', m, m)
    X = BlockMatrix([[A, B], [C, D]])
    assert block_collapse(X.inverse()) == BlockMatrix([
        [(A - B * D.I * C).I, -(A - B * D.I * C).I * B * D.I],
        [-D.I * C * (A - B * D.I * C).I, D.I + D.I * C * (A - B * D.I * C).I * B * D.I],
    ])


def test_BlockMatrix_2x2_inverse_numeric():
    """Test 2x2 block matrix inversion numerically for all 4 formulas"""
    M = Matrix([[1, 2], [3, 4]])
    # rank deficient matrices that have full rank when two of them combined
    D1 = Matrix([[1, 2], [2, 4]])
    D2 = Matrix([[1, 3], [3, 9]])
    D3 = Matrix([[1, 4], [4, 16]])
    assert D1.rank() == D2.rank() == D3.rank() == 1
    assert (D1 + D2).rank() == (D2 + D3).rank() == (D3 + D1).rank() == 2

    # Only A is invertible
    K = BlockMatrix([[M, D1], [D2, D3]])
    assert block_collapse(K.inv()).as_explicit() == K.as_explicit().inv()
    # Only B is invertible
    K = BlockMatrix([[D1, M], [D2, D3]])
    assert block_collapse(K.inv()).as_explicit() == K.as_explicit().inv()
    # Only C is invertible
    K = BlockMatrix([[D1, D2], [M, D3]])
    assert block_collapse(K.inv()).as_explicit() == K.as_explicit().inv()
    # Only D is invertible
    K = BlockMatrix([[D1, D2], [D3, M]])
    assert block_collapse(K.inv()).as_explicit() == K.as_explicit().inv()


@slow
def test_BlockMatrix_3x3_symbolic():
    # Only test one of these, instead of all permutations, because it's slow
    rowblocksizes = (n, m, k)
    colblocksizes = (m, k, n)
    K = BlockMatrix([
        [MatrixSymbol('M%s%s' % (rows, cols), rows, cols) for cols in colblocksizes]
        for rows in rowblocksizes
    ])
    collapse = block_collapse(K.I)
    assert isinstance(collapse, BlockMatrix)


def test_BlockDiagMatrix():
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', m, m)
    C = MatrixSymbol('C', l, l)
    M = MatrixSymbol('M', n + m + l, n + m + l)

    X = BlockDiagMatrix(A, B, C)
    Y = BlockDiagMatrix(A, 2*B, 3*C)

    assert X.blocks[1, 1] == B
    assert X.shape == (n + m + l, n + m + l)
    assert all(X.blocks[i, j].is_ZeroMatrix if i != j else X.blocks[i, j] in [A, B, C]
            for i in range(3) for j in range(3))
    assert X.__class__(*X.args) == X
    assert X.get_diag_blocks() == (A, B, C)

    assert isinstance(block_collapse(X.I * X), Identity)

    assert bc_matmul(X*X) == BlockDiagMatrix(A*A, B*B, C*C)
    assert block_collapse(X*X) == BlockDiagMatrix(A*A, B*B, C*C)
    #XXX: should be == ??
    assert block_collapse(X + X).equals(BlockDiagMatrix(2*A, 2*B, 2*C))
    assert block_collapse(X*Y) == BlockDiagMatrix(A*A, 2*B*B, 3*C*C)
    assert block_collapse(X + Y) == BlockDiagMatrix(2*A, 3*B, 4*C)

    # Ensure that BlockDiagMatrices can still interact with normal MatrixExprs
    assert (X*(2*M)).is_MatMul
    assert (X + (2*M)).is_MatAdd

    assert (X._blockmul(M)).is_MatMul
    assert (X._blockadd(M)).is_MatAdd

def test_BlockDiagMatrix_nonsquare():
    A = MatrixSymbol('A', n, m)
    B = MatrixSymbol('B', k, l)
    X = BlockDiagMatrix(A, B)
    assert X.shape == (n + k, m + l)
    assert X.shape == (n + k, m + l)
    assert X.rowblocksizes == [n, k]
    assert X.colblocksizes == [m, l]
    C = MatrixSymbol('C', n, m)
    D = MatrixSymbol('D', k, l)
    Y = BlockDiagMatrix(C, D)
    assert block_collapse(X + Y) == BlockDiagMatrix(A + C, B + D)
    assert block_collapse(X * Y.T) == BlockDiagMatrix(A * C.T, B * D.T)
    raises(NonInvertibleMatrixError, lambda: BlockDiagMatrix(A, C.T).inverse())

def test_BlockDiagMatrix_determinant():
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', m, m)
    assert det(BlockDiagMatrix()) == 1
    assert det(BlockDiagMatrix(A)) == det(A)
    assert det(BlockDiagMatrix(A, B)) == det(A) * det(B)

    # non-square blocks
    C = MatrixSymbol('C', m, n)
    D = MatrixSymbol('D', n, m)
    assert det(BlockDiagMatrix(C, D)) == 0

def test_BlockDiagMatrix_trace():
    assert trace(BlockDiagMatrix()) == 0
    assert trace(BlockDiagMatrix(ZeroMatrix(n, n))) == 0
    A = MatrixSymbol('A', n, n)
    assert trace(BlockDiagMatrix(A)) == trace(A)
    B = MatrixSymbol('B', m, m)
    assert trace(BlockDiagMatrix(A, B)) == trace(A) + trace(B)

    # non-square blocks
    C = MatrixSymbol('C', m, n)
    D = MatrixSymbol('D', n, m)
    assert isinstance(trace(BlockDiagMatrix(C, D)), Trace)

def test_BlockDiagMatrix_transpose():
    A = MatrixSymbol('A', n, m)
    B = MatrixSymbol('B', k, l)
    assert transpose(BlockDiagMatrix()) == BlockDiagMatrix()
    assert transpose(BlockDiagMatrix(A)) == BlockDiagMatrix(A.T)
    assert transpose(BlockDiagMatrix(A, B)) == BlockDiagMatrix(A.T, B.T)

def test_issue_2460():
    bdm1 = BlockDiagMatrix(Matrix([i]), Matrix([j]))
    bdm2 = BlockDiagMatrix(Matrix([k]), Matrix([l]))
    assert block_collapse(bdm1 + bdm2) == BlockDiagMatrix(Matrix([i + k]), Matrix([j + l]))

def test_blockcut():
    A = MatrixSymbol('A', n, m)
    B = blockcut(A, (n/2, n/2), (m/2, m/2))
    assert B == BlockMatrix([[A[:n/2, :m/2], A[:n/2, m/2:]],
                             [A[n/2:, :m/2], A[n/2:, m/2:]]])

    M = ImmutableMatrix(4, 4, range(16))
    B = blockcut(M, (2, 2), (2, 2))
    assert M == ImmutableMatrix(B)

    B = blockcut(M, (1, 3), (2, 2))
    assert ImmutableMatrix(B.blocks[0, 1]) == ImmutableMatrix([[2, 3]])

def test_reblock_2x2():
    B = BlockMatrix([[MatrixSymbol('A_%d%d'%(i,j), 2, 2)
                            for j in range(3)]
                            for i in range(3)])
    assert B.blocks.shape == (3, 3)

    BB = reblock_2x2(B)
    assert BB.blocks.shape == (2, 2)

    assert B.shape == BB.shape
    assert B.as_explicit() == BB.as_explicit()

def test_deblock():
    B = BlockMatrix([[MatrixSymbol('A_%d%d'%(i,j), n, n)
                    for j in range(4)]
                    for i in range(4)])

    assert deblock(reblock_2x2(B)) == B

def test_block_collapse_type():
    bm1 = BlockDiagMatrix(ImmutableMatrix([1]), ImmutableMatrix([2]))
    bm2 = BlockDiagMatrix(ImmutableMatrix([3]), ImmutableMatrix([4]))

    assert bm1.T.__class__ == BlockDiagMatrix
    assert block_collapse(bm1 - bm2).__class__ == BlockDiagMatrix
    assert block_collapse(Inverse(bm1)).__class__ == BlockDiagMatrix
    assert block_collapse(Transpose(bm1)).__class__ == BlockDiagMatrix
    assert bc_transpose(Transpose(bm1)).__class__ == BlockDiagMatrix
    assert bc_inverse(Inverse(bm1)).__class__ == BlockDiagMatrix

def test_invalid_block_matrix():
    raises(ValueError, lambda: BlockMatrix([
        [Identity(2), Identity(5)],
    ]))
    raises(ValueError, lambda: BlockMatrix([
        [Identity(n), Identity(m)],
    ]))
    raises(ValueError, lambda: BlockMatrix([
        [ZeroMatrix(n, n), ZeroMatrix(n, n)],
        [ZeroMatrix(n, n - 1), ZeroMatrix(n, n + 1)],
    ]))
    raises(ValueError, lambda: BlockMatrix([
        [ZeroMatrix(n - 1, n), ZeroMatrix(n, n)],
        [ZeroMatrix(n + 1, n), ZeroMatrix(n, n)],
    ]))
