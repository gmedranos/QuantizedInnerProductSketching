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
        rng = np.random.default_rng(seed = self.seed)
        for i in range(0, self.sketch_size):
            R = rng.normal(0, 1, dim)
            vec.append(R.T @ vector)
        
        vec = np.array(vec)

        return SHSketch(np.linalg.norm(vector), np.sign(vec), self.sketch_size)
    
    # Get a list of vectors to sketch
    def batch_sketch(self, vectors):
        dim = len(vectors[0])
        rng = np.random.default_rng(seed=self.seed)
        num_vectors = len(vectors)
        vector_matrix = np.array(vectors)
        sh_matrix = []

        for i in range(0, self.sketch_size):
            R = rng.normal(0, 1, dim)
            sh = vector_matrix @ R
            sh_matrix.append(np.sign(sh))
        
        lists = list(np.array(sh_matrix).T)

        list_of_sk = []
        count = 0
        for i in lists:
            list_of_sk.append(SHSketch(np.linalg.norm(vectors[count]), i, self.sketch_size))
            count += 1
        
        return list_of_sk
    