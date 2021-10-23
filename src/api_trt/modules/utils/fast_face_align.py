# Copy of https://github.com/deepinsight/insightface/tree/master/python-package/insightface/utils/face_align.py
import cv2
import numpy as np
import math

import time
import numba as nb

src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
                 [51.157, 89.050], [57.025, 89.702]],
                dtype=np.float32)
#<--left
src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
                 [45.177, 86.190], [64.246, 86.758]],
                dtype=np.float32)

#---frontal
src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                 [42.463, 87.010], [69.537, 87.010]],
                dtype=np.float32)

#-->right
src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
                 [48.167, 86.758], [67.236, 86.190]],
                dtype=np.float32)

#-->right profile
src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
                 [55.388, 89.702], [61.257, 89.050]],
                dtype=np.float32)

src = np.array([src1, src2, src3, src4, src5])
src_map = {112: src, 224: src * 2}

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

arcface_src = np.expand_dims(arcface_src, axis=0)


@nb.njit
def np_apply_along_axis(func1d, axis, arr):
  assert arr.ndim == 2
  assert axis in [0, 1]
  if axis == 0:
    result = np.empty(arr.shape[1])
    for i in range(len(result)):
      result[i] = func1d(arr[:, i])
  else:
    result = np.empty(arr.shape[0])
    for i in range(len(result)):
      result[i] = func1d(arr[i, :])

  return result

@nb.njit
def np_mean(array, axis):
  return np_apply_along_axis(np.mean, axis, array)

@nb.njit
def np_std(array, axis):
  return np_apply_along_axis(np.std, axis, array)

@nb.njit
def np_var(array,axis):
    return np_apply_along_axis(np.var,axis,array)

@nb.njit(fastmath=True)
def _umeyama(src, dst, estimate_scale):

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = np_mean(src,0)
    dst_mean = np_mean(dst, 0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = dst_demean.T @ src_demean / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V

    scale = 1.0
    if estimate_scale:
        # Eq. (41) and (42).
        div = np_var(src_demean,0)
        div = np.sum(div)
        scale = scale / div * (S @ d)

    T[:dim, dim] = dst_mean - scale * (np.ascontiguousarray(T[:dim, :dim]) @ np.ascontiguousarray(src_mean.T))
    T[:dim, :dim] *= scale

    return T


# lmk is prediction; src is template
@nb.njit()
def estimate_norm(lmk, image_size=112, mode='arcface'):
    assert lmk.shape == (5, 2)

    lmk_tran = np.ones((5, 3))
    lmk_tran[:, :2] = lmk

    #lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_error = np.inf

    assert image_size == 112
    src = arcface_src

    params = _umeyama(lmk, src[0],True)
    #took = time.time() - t0
    #print(f"Est TFORM: {took * 1000:.3f} ms.")
    M = params[0:2, :]
    return M

@nb.njit
def estimate_norm_batch(lmks, image_size=112, mode='arcface'):
    Ms = []
    for lmk in lmks:
        Ms.append(estimate_norm(lmk, image_size, mode))
    return Ms

def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped

def norm_crop_batched(img, landmarks, image_size=112, mode='arcface'):
    Ms = estimate_norm_batch(landmarks, image_size, mode)
    crops = []
    for M in Ms:
        warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
        crops.append(warped)
    return crops
