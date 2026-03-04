import numpy as np
from abc import ABC, abstractmethod
from .abstract_class import InnerProdSketcher, InnerProdSketch, hash_kwise
from scipy.sparse import vstack
import scipy as sp

class SHSketch(InnerProdSketch):
    def __init__(self, norm, signs, sketch_size, len = 1, seed = 1):
        self.norm = norm
        self.signs = signs
        self.sketch_size = sketch_size
        self.len = len
        self.seed = seed
        self.matrix = None

    def inner_product(self, other):
        min_size = min(len(self.signs), len(other.signs))

        if(self.sketch_size != 0):
            sum = np.sum(other.signs[:min_size] == self.signs[:min_size])
            return self.norm * other.norm * np.cos((1 - sum / min_size) * np.pi)
        return 0
    
    def QJL(self, vector):
        rng = np.random.default_rng(seed = self.seed)
        if self.sketch_size == 0:
            return 0
        if(self.matrix is None):
            Sq = []
            for j in range(0, self.sketch_size):
                R = rng.normal(0, 1, len(vector))                
                Sq.append(vector @ R)
            
            Sq = np.array(Sq)
            return np.sqrt(np.pi / 2) / self.sketch_size * self.norm * (Sq @ self.signs)

        return np.sqrt(np.pi / 2) / self.sketch_size * self.norm * ((self.matrix @ vector) @ self.signs)

    def get_vector(self):
        angles = np.zeros(self.len)
        rng = np.random.default_rng(seed = self.seed)
        
        for j in range(0, self.sketch_size):
            R = rng.normal(0, 1, self.len)
            signs = np.sign(R)
            angles = angles + (signs == self.signs[j]).astype(int)
        

        return self.norm * np.cos((1 - angles / self.sketch_size) * np.pi)
    
    def get_norm(self):
        return self.norm

    def set_matrix(self, matrix):
        self.matrix = matrix

class SH(InnerProdSketcher):
    def __init__(self, sketch_size, seed, matrix=None):
        self.sketch_size = sketch_size
        self.seed = seed
        self.matrix = matrix

    def sketch(self, vector):
        dim = vector.shape[0]
        vec = []
        rng = np.random.default_rng(self.seed)

        if(self.matrix is None):
            for i in range(0, self.sketch_size):
                R = rng.normal(0, 1, dim)
                vec.append(vector.dot(R))
            
            vec = np.array(vec)
        
        else:
            vec = self.matrix @ vector

        return SHSketch(np.linalg.norm(vector), np.sign(vec), self.sketch_size, dim, self.seed)
    
    def sketch_all(self, vectors, batch_size = 50000):
        list_of_sk = []

        for i in range(0, (len(vectors) - 1) // batch_size + 1):
            list_of_sk = list_of_sk + self.batch_sketch(vectors[i*batch_size : (i + 1) *batch_size])
        return list_of_sk

    # Get a list of vectors to sketch
    def batch_sketch(self, vectors):
        vectors = [v.reshape(1, -1) for v in vectors]
        dim = vectors[0].shape[1]
        rng = np.random.default_rng(seed=self.seed)
        num_vectors = len(vectors)
        vector_matrix = vstack(vectors)
        sh_matrix = []

        for i in range(0, self.sketch_size):
            R = rng.normal(0, 1, dim)
            sh = vector_matrix @ R
            sh_matrix.append(np.sign(sh))
        
        lists = list(np.array(sh_matrix).T)

        list_of_sk = []
        count = 0
        for i in lists:
            list_of_sk.append(SHSketch(np.linalg.norm(vectors[count].toarray()), np.array(i, dtype=np.int8), self.sketch_size))
            count += 1
        
        return list_of_sk
    
    def batch_sketch2(self, vectors):
        dim = len(vectors[0])        
        rng = np.random.default_rng(seed=self.seed)
        num_vectors = len(vectors)
        vector_matrix = np.array(vectors)
        sh_matrix = []
        gauss_matrix = []

        for i in range(0, self.sketch_size):
            R = rng.normal(0, 1, dim)
            gauss_matrix.append(R)
            sh = vector_matrix @ R
            sh_matrix.append(np.sign(sh))
        
        lists = list(np.array(sh_matrix).T)

        list_of_sk = []
        count = 0
        gauss_matrix = np.array(gauss_matrix)
        for i in lists:
            list_of_sk.append(SHSketch(np.linalg.norm(vectors[count]), np.array(i, dtype=np.int8), self.sketch_size))
            list_of_sk[-1].set_matrix(gauss_matrix)
            count += 1
        
        return list_of_sk