import numpy as np
from abc import ABC, abstractmethod
from .abstract_class import InnerProdSketcher, InnerProdSketch, hash_kwise
from scipy.optimize import fsolve
from .priority_sampling import PSSketch, PS
from scipy.sparse import coo_array

from collections import defaultdict

import numpy as np

class ASASSketch(InnerProdSketch):
    def __init__(self, head_vec, tail_sketch, seed, sketcher_class_tail, tail_size, true_tail):
        self.head_vec = head_vec
        self.tail_sketch = tail_sketch
        self.sketcher_class_tail = sketcher_class_tail
        self.seed = seed
        self.tail_size = tail_size
        self.true_tail = true_tail

    def inner_product(self, other, matrix=None):
        sketcher = self.sketcher_class_tail(self.tail_size, self.seed)
        
        head_vec = self.head_vec.astype(np.float32).toarray()
        other_head_vec = other.head_vec.astype(np.float32).toarray()
        idxs = np.where((head_vec != 0) & (other_head_vec != 0))[0]

        W1, W2, W3, W4 = 0, 0, 0, 0

        W1 = np.float64(head_vec) @ np.float64(other_head_vec)

        self.tail_sketch.set_matrix(matrix)
        other.tail_sketch.set_matrix(matrix)

        other_sampled_overlap = np.copy(other_head_vec)
        other_sampled_overlap[idxs] = 0 

        self_sampled_overlap = np.copy(head_vec)
        self_sampled_overlap[idxs] = 0 

        W2 = self.tail_sketch.inner_product_asym(other_sampled_overlap)
        W3 = other.tail_sketch.inner_product_asym(self_sampled_overlap)

        self.tail_sketch.set_matrix(None)
        other.tail_sketch.set_matrix(None)
            
        W4 = self.tail_sketch.inner_product(other.tail_sketch)


        #print(compute_var(self.head_vec, self.true_tail, other.head_vec, other.true_tail, self.tail_size, other.tail_size))
        #print(W1, " ", W2, " ", W3, " ", W4)
        #print(self.head_vec @ other.head_vec, " ", self.true_tail @ other.head_vec, " ", other.true_tail @ self.head_vec, " ", self.true_tail @ other.true_tail)

        return W1 + W2 + W3 + W4
    
    def inner_product_asym(self, vec, matrix=None):
        head_vec = self.head_vec.astype(np.float32).toarray()
        self.tail_sketch.set_matrix(matrix)

        W1 = head_vec @ vec
        vec_to_sketch = np.zeros(len(vec))
        vec_to_sketch[head_vec == 0] = vec[head_vec == 0]
        W2 = self.tail_sketch.inner_product_asym(vec_to_sketch)

        self.tail_sketch.set_matrix(None)
        return W1 + W2
    
    '''
    def inner_product_asym(self, vec, matrix=None):
        head_vec = self.head_vec.astype(np.float32).toarray()
        sketcher = self.sketcher_class_tail(self.tail_size, self.seed)
        sketcher.set_matrix(matrix)

        W1 = head_vec @ vec
        vec_to_sketch = np.zeros(len(vec))
        vec_to_sketch[head_vec == 0] = vec[head_vec == 0]
        W2 = self.tail_sketch.inner_product(sketcher.sketch(vec_to_sketch))

        sketcher.set_matrix(None)
        return W1 + W2
    '''


def compute_var(ah, at, bh, bt, ska, skb):
    norm_ah = np.linalg.norm(ah)**2
    norm_at = np.linalg.norm(at)**2
    norm_bh = np.linalg.norm(bh)**2
    norm_bt = np.linalg.norm(bt)**2
    return (norm_ah * norm_bt)/ skb + (norm_at * norm_bh) / skb + (norm_at * norm_bt) / min(ska, skb)

def heuristic_formula(vec_head, vec_tail, budget, mult_factor=1):
    norm_head = np.linalg.norm(vec_head)**2
    norm_tail = np.linalg.norm(vec_tail)**2

    return (mult_factor * norm_head * norm_tail + norm_tail**2) / budget

def adaptive_select_old(vec, max_budget, mult_factor=1):
    vec_norm = vec / np.linalg.norm(vec)
    bits_per_key = 16 + np.ceil(np.log2(len(vec)))
    idxs_sort = np.argsort(np.abs(vec_norm))
    # Largest to smallest
    idxs_sort = np.flip(idxs_sort)

    counter = 0
    vec_head = np.zeros(len(vec))
    vec_tail = vec

    next_head = vec_head.copy()
    next_head[idxs_sort[counter]] = vec[idxs_sort[counter]]
    next_tail = vec - next_head



    while(heuristic_formula(vec_head, vec_tail, max_budget, mult_factor) > heuristic_formula(next_head, next_tail, max_budget - bits_per_key, mult_factor) and counter < len(vec) - 1 and max_budget - bits_per_key > 0):
        counter += 1
        max_budget -= bits_per_key
        vec_head = next_head.copy()
        vec_tail = next_tail.copy()
        next_head[idxs_sort[counter]] = vec[idxs_sort[counter]]
        next_tail = vec - next_head

    return idxs_sort[:counter]

def heuristic_formula_asym(vec_tail, budget):
    norm_tail = np.linalg.norm(vec_tail)
    return  (norm_tail**2) / budget

def adaptive_select_asym(vec, max_budget):
    vec_norm = vec / np.linalg.norm(vec)
    bits_per_key = 16 + np.ceil(np.log2(len(vec)))
    idxs_sort = np.argsort(np.abs(vec_norm))
    # Largest to smallest
    idxs_sort = np.flip(idxs_sort)

    counter = 0
    vec_head = np.zeros(len(vec))
    vec_tail = vec

    next_head = vec_head.copy()
    next_head[idxs_sort[counter]] = vec[idxs_sort[counter]]
    next_tail = vec - next_head

    while(heuristic_formula_asym(vec_tail, max_budget) > heuristic_formula_asym(next_tail, max_budget - bits_per_key) and counter < len(vec) - 1 and max_budget - bits_per_key > 0):
        counter += 1
        max_budget -= bits_per_key
        vec_head = next_head.copy()
        vec_tail = next_tail.copy()
        next_head[idxs_sort[counter]] = vec[idxs_sort[counter]]
        next_tail = vec - next_head

    return idxs_sort[:counter]


def adaptive_select(vec, max_budget, mult_factor=1.0):
    vec = np.asarray(vec)
    d = len(vec)
    
    # Calculate memory cost per key
    bits_per_key = 16 + np.ceil(np.log2(d))
    
    # Precompute squared magnitudes (this avoids calculating norms in the loop)
    squared_vals = vec**2
    
    # Sort indices by magnitude, largest to smallest
    # (Note: sorting vec vs vec_norm yields the exact same order, so we skip normalization)
    idxs_sort = np.argsort(squared_vals)[::-1]
    
    # Initialize energies (head starts empty, tail has everything)
    norm_head_sq = 0.0
    norm_tail_sq = np.sum(squared_vals)
    
    current_budget = max_budget
    counter = 0
    
    # Helper function to compute the heuristic score using scalar energies
    def calc_score(h_sq, t_sq, budget):
        return (mult_factor * h_sq * t_sq + t_sq**2) / budget

    # Greedily evaluate elements
    for i in range(d):
        next_budget = current_budget - bits_per_key
        
        # Stop if we run out of budget for another exact entry
        if next_budget <= 0:
            break
            
        # The energy of the candidate element to move from tail to head
        x_i_sq = squared_vals[idxs_sort[i]]
        
        # Calculate what the energies WOULD be if we move this element
        next_head_sq = norm_head_sq + x_i_sq
        next_tail_sq = norm_tail_sq - x_i_sq
        
        # Avoid floating point underflow issues for tail energy
        if next_tail_sq < 1e-12:
            next_tail_sq = 0.0

        current_score = calc_score(norm_head_sq, norm_tail_sq, current_budget)
        next_score = calc_score(next_head_sq, next_tail_sq, next_budget)
        
        # If moving the element decreases the error bound, keep it!
        if next_score < current_score:
            counter += 1
            current_budget = next_budget
            norm_head_sq = next_head_sq
            norm_tail_sq = next_tail_sq
        else:
            # Because elements are sorted descending, if this large one doesn't help, 
            # the subsequent smaller ones won't either. We can safely stop.
            break
            
    return idxs_sort[:counter]

class ASaS(InnerProdSketcher):
    def __init__(self, size, seed: int, tail_sketch_class, dim=None) -> None:
        self.size = size
        self.seed: int = seed
        self.tail_sketch_class = tail_sketch_class
        self.matrix = None

        if(dim is not None):
            # Get the random matrix to make QJL faster
            rng = np.random.default_rng(seed = self.seed)
            self.matrix = rng.normal(0, 1, (size, dim))

    def sketch(self, vector : np.ndarray):
        if (type(vector) == coo_array):
            dim = vector.shape[0]
            vector = vector.toarray()
        else:
            dim = len(vector)
        
        idxs = np.arange(dim)

        #idxs_big = select_exact_indices(vector, self.size, (16 + np.ceil(np.log2(dim))), 1)
        idxs_big = adaptive_select_asym(vector, self.size)
        idxs_small = np.setdiff1d(idxs, idxs_big)

        tail_size = int(self.size - len(idxs_big) * (16 + np.ceil(np.log2(dim))))

        vector_head = np.zeros_like(vector)
        vector_tail = np.zeros_like(vector)

        vector_head[idxs_big] = vector[idxs_big]
        
        # Use float16 here
        vector_head = np.float16(vector_head)
        vector_tail[idxs_small] = vector[idxs_small]

        if(self.matrix is not None):        
            sketcher_small = self.tail_sketch_class(tail_size, self.seed, self.matrix[:tail_size])
        else:
            sketcher_small = self.tail_sketch_class(tail_size, self.seed)
        sketch_small = sketcher_small.sketch(vector_tail)
        

        #Have to upscale to float 32 here because scipy does not allow for sparse vectors with float16
        #Note that this does not change the value so the experiments are still the same if we kept float16
        vector_head = np.float32(vector_head)

        #print(idxs_big)
        #print(compute_var(vector_head, vector_tail, vector_head, vector_tail, tail_size, tail_size))
        return ASASSketch(coo_array(vector_head), sketch_small, self.seed, self.tail_sketch_class, tail_size, vector_tail)
    
    def sketch_asym(self, vector : np.ndarray):
        if (type(vector) == coo_array):
            dim = vector.shape[0]
            vector = vector.toarray()
        else:
            dim = len(vector)
        
        idxs = np.arange(dim)

        #idxs_big = select_exact_indices(vector, self.size, (16 + np.ceil(np.log2(dim))), 1)
        idxs_big = adaptive_select_asym(vector, self.size)
        idxs_small = np.setdiff1d(idxs, idxs_big)

        tail_size = int(self.size - len(idxs_big) * (16 + np.ceil(np.log2(dim))))

        vector_head = np.zeros_like(vector)
        vector_tail = np.zeros_like(vector)

        vector_head[idxs_big] = np.float16(vector[idxs_big])
        vector_tail[idxs_small] = vector[idxs_small].copy()

        if(self.matrix is not None):        
            sketcher_small = self.tail_sketch_class(tail_size, self.seed, self.matrix[:tail_size])
        else:
            sketcher_small = self.tail_sketch_class(tail_size, self.seed)
        sketch_small = sketcher_small.sketch(vector_tail)
        
        #print(idxs_big)
        #print(compute_var(vector_head, vector_tail, vector_head, vector_tail, tail_size, tail_size))
        return ASASSketch(coo_array(vector_head), sketch_small, self.seed, self.tail_sketch_class, tail_size, vector_tail)

    def get_matrix(self):
        return self.matrix
