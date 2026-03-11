from libero.libero import benchmark
import numpy as np
import time



benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict['libero_spatial']()
num_tasks = task_suite.n_tasks

print('try')