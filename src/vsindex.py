class VSIndex():
    def __init__(self):
        self.vectors = []

    def insert_vectors(self, vectors):
        for i in vectors:
            self.vectors.append(i)
    
    def query(self, q, k=1):
        sorted_vec = sorted(self.vectors, key= lambda x: q.inner_product(x[1]), reverse=True)
        return sorted_vec[:k]

