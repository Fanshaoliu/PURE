# 保存为 test_ray.py
import ray

ray.init()

@ray.remote
def f(x):
    return x + 1

print(ray.get([f.remote(i) for i in range(5)]))
ray.shutdown()
