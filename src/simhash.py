import numpy as np
from abc import ABC, abstractmethod
from .abstract_class import InnerProdSketcher, InnerProdSketch, hash_kwise

class SHSketch(InnerProdSketch):
    def __init__(self, norm, signs, sketch_size):
        self.norm = norm
        self.signs = signs
        self.sketch_size = sketch_size

    def inner_product(self, other):
        sum = np.sum(other.signs == self.signs)
        return self.norm * other.norm * np.cos((1 - sum / self.sketch_size) * np.pi)

class SH(InnerProdSketcher):
    def __init__(self, sketch_size, seed):
        self.sketch_size = sketch_size
        self.seed = seed

    def sketch(self, vector):
        dim = len(vector)
        vec = []
        for i in range(0, self.sketch_size):
            np.random.seed(self.seed + i)
            R = np.random.normal(0, 1, dim)
            vec.append(R.T @ vector)
        
        vec = np.array(vec)

        return SHSketch(np.linalg.norm(vector), np.sign(vec), self.sketch_size)
    