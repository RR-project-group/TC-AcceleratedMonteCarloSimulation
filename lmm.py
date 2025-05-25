import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import pandas as pd

def generate_correlation_matrix(N, rho=0.5):
    corr = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            corr[i, j] = rho ** abs(i - j)
    return corr

def sample_normals(shape, dtype):
    samples = tf.random.normal(shape, dtype=tf.float32)
    return tf.cast(samples, dtype)

@tf.function
def simulate_lmm_paths(cov, paths, T, N, dtype):
    cov_float32 = tf.cast(cov, tf.float32)
    chol = tf.linalg.cholesky(cov_float32)
    if dtype == tf.bfloat16:
        chol = tf.cast(chol, tf.bfloat16)

    normals = sample_normals((paths, T, N), dtype)
    libor_t = tf.ones((paths, N), dtype=dtype)
    paths_arr = []
    for t in range(T):
        z = tf.matmul(normals[:, t, :], chol, transpose_b=True)
        libor_t = libor_t * tf.exp(-0.5 * 0.01 + 0.01 * z)
        paths_arr.append(libor_t)
    return tf.stack(paths_arr, axis=1)
