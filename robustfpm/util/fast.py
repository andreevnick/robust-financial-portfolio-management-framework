import numpy as np
import numba as nb

def rowsort(a):
    if rowsorted(a):
        return a
    a = a[np.argsort(a[:,1])]
    a = a[np.argsort(a[:,0], kind='stable')]
    return a

@nb.njit()
def rowsorted(a):
    n, m = a.shape
    assert m == 2
    for i in range(1, n):
        if (merge_gt(a[i - 1], a[i])):
            return False
    return True


@nb.njit()
def merge_gt(a, b):
    return (a[0] > b[0]) or (a[0] == b[0]) and (a[1] > b[1])

@nb.njit()
def merge_ne(a, b):
    return (a[0] != b[0]) or (a[1] != b[1])

@nb.njit()
def merge_into(A, B):
    An, Am = A.shape
    assert Am == 2
    Bn, Bm = B.shape
    assert Bm == 2
    res = np.zeros((An + Bn, 2), A.dtype)
    i = 0
    j = 0
    k = 0
    for p in range(An + Bn):
        #print(p, k, i, j, A[i] if i < An else None, B[j] if j < Bn else None)
        if (j >= Bn) or ((i < An) and not(merge_gt(A[i], B[j]))):
            # if (k == 0) or merge_ne(res[k - 1], A[i]):
            #     res[k] = A[i]
            #     k += 1
            res[k] = A[i]
            k += 1
            i += 1
        else:
            if (k == 0) or merge_ne(res[k - 1], B[j]):
                res[k] = B[j]
                k += 1
            j += 1
    return res[:k]

def unique_points_union_fast(points1, points2):
    points2 = rowsort(points2)

    return merge_into(points1, points2)
