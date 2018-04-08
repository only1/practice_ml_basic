import tensorflow as tf
import numpy as np

t = np.array([[[0, 1, 2],
              [3, 4, 5],
              [6, 7, 8],
              [9, 10, 11]],
              [[20, 21, 22],
               [23, 24, 25],
               [26, 27, 28],
               [29, 30, 31]]])

print(t.shape)
print(t.argmax(axis=0))
print(t.argmax(axis=1))
print(t.argmax(axis=2))
