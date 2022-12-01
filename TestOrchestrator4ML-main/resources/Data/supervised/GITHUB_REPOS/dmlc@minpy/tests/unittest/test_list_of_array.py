import minpy.numpy as np
import minpy.numpy.random as rnd


def test_stack():
    arr = [rnd.randn(3, 4) for _ in range(10)]
    res = np.stack(arr)
    assert res.shape == (10, 3, 4)

def test_concatenate():
    arr = [rnd.randn(3, 4) for _ in range(10)]
    res = np.concatenate(arr, axis=1)
    assert res.shape == (3, 40)

if __name__  == '__main__':
    test_stack()
    test_concatenate()
