
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import pandas as pd

# ===== Monte Carlo simulation =====
@tf.function
def monte_carlo_bs_tpu(S0, K, T, r, sigma, n_paths, n_steps, dtype=tf.float32, seed=(0,0)):
    dt = T / n_steps
    # generate random numbers
    z = tf.random.stateless_normal(shape=(n_paths, n_steps), seed=seed, dtype=dtype)
    # generate log returns
    log_returns = ((r - 0.5 * sigma**2) * dt + sigma * tf.sqrt(dt) * z)
    log_paths = tf.cumsum(log_returns, axis=1)
    ST = S0 * tf.exp(log_paths[:, -1])  # final price
    payoff = tf.nn.relu(ST - K)
    price = tf.exp(-r * T) * tf.reduce_mean(payoff)
    return price

def run_bs_simulation(n_paths, n_steps, dtype=tf.float32, seed=(42, 42)):
    S0 = tf.constant(100.0, dtype=dtype)
    K = tf.constant(120.0, dtype=dtype)
    T = tf.constant(1.0, dtype=dtype)
    r = tf.constant(0.05, dtype=dtype)
    sigma = tf.constant(0.2, dtype=dtype)

    return tf.cast(monte_carlo_bs_tpu(S0, K, T, r, sigma, n_paths, n_steps, dtype=dtype, seed=seed), tf.float32)

