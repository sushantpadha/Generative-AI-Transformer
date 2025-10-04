import tensorflow as tf
import time

# Function to run matrix multiplication on a specified device
def compute_on_device(device_name):
    print(f"\nRunning on {device_name}")
    with tf.device(device_name):
        start = time.time()
        # Generate random matrices
        a = tf.random.normal([10000, 10000])
        b = tf.random.normal([10000, 10000])
        # Perform matrix multiplication
        c = tf.matmul(a, b)
        end = time.time()
    print(f"Computation time on {device_name}: {end - start:.4f} seconds")

# Run on GPU
compute_on_device('/GPU:0')

# Run on CPU
compute_on_device('/CPU:0')

