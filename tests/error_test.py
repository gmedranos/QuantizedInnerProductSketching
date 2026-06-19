import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_array

class VSIndex():
    def __init__(self):
        self.vectors = []

    def insert_vectors(self, vectors):
        for i in vectors:
            self.vectors.append(i)
    
    def query(self, q, k=1):
        sorted_vec = sorted(self.vectors, key= lambda x: q.inner_product(x[1]), reverse=True)
        return sorted_vec[:k]
    
    def query_matrix(self, q, matrix, k=1):
        sorted_vec = sorted(self.vectors, key= lambda x: q.inner_product(x[1], matrix), reverse=True)
        return sorted_vec[:k]
    
    def query_dist(self, q, k=1):
        sorted_vec = sorted(self.vectors, key= lambda x: x[1].get_norm() ** 2 + q.get_norm() ** 2 - 2 * q.inner_product(x[1]))
        return sorted_vec[:k]
    
class StandardVec():
    def __init__(self, vec):
        self.vec = vec
    def inner_product(self, other):
        if(type(self.vec) == coo_array):

            return other.vec.toarray().dot(self.vec.toarray())
        else:
            return self.vec @ other.vec
        
def test_sketchers(list_sketcher_classes, list_sizes, trials, list_of_list_of_vecs, queries, list_names, title, markers=None, colors=None):

    errors = []
    for sketcher_class in list_sketcher_classes:
        errors_class = []
        for size in list_sizes:
            error = 0
            for seed in range(0, trials):
                    sketcher = sketcher_class(size, seed)
                    matrix = sketcher.get_matrix()
                    for j in range(0, len(queries)):
                        if(type(queries[j]) == coo_array):
                            query = queries[j].toarray()
                        else:
                            query = queries[j]
                        q_sketch = sketcher.sketch(query)
                        vecs = list_of_list_of_vecs[j]
                        for vec in vecs:
                            vec_sketch = sketcher.sketch(vec)
                            error += np.abs(vec_sketch.inner_product(q_sketch, matrix) - vec @ query) / (np.linalg.norm(vec) * np.linalg.norm(query))
            errors_class.append(error / (len(queries) *  len(vecs) * trials))
        
        errors.append(errors_class)
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times Roman"],
        "font.size": 20
    })

    if (colors is None):
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(errors)))

    if markers is None:
        markers = ['x' for i in range(0, len(errors))]

    for i in range(0, len(errors)):
        plt.plot(list_sizes, errors[i], marker=markers[i], color=colors[i], label=list_names[i], linewidth=2)


    #plt.title(title)
    plt.xlabel("Sketch size")
    plt.ylabel("Error")
    #plt.ylim(0, 0.1)
    #plt.legend()
    plt.grid(False)
    plt.tight_layout()

    handles, labels = plt.gca().get_legend_handles_labels()

    plt.savefig("figs/" + title + ".pdf", format="pdf", bbox_inches='tight')
    # Show the plot
    plt.show()
    plt.close()

    fig_leg = plt.figure(figsize=(3,1))
    fig_leg.legend(handles=handles, labels=labels, loc='center', ncol=3)
    plt.savefig("figs/" + title + "_legend.pdf", format="pdf", bbox_inches='tight')
    plt.close()
    
    
    return errors

def get_top_k(vectors, q, k):
    vsindex = VSIndex()
    numbered_vectors = [(i, StandardVec(vectors[i])) for i in range(0, len(vectors))]
    vsindex.insert_vectors(numbered_vectors)

    answer = vsindex.query(StandardVec(q), k)
    #Might have to change this .toarray() later
    if(type(answer[0][1].vec) == coo_array):
        return [i[1].vec.toarray() for i in answer]
    else:
        return [i[1].vec for i in answer]

def test_top_k_sketches(list_sketcher_classes, list_sizes, trials, vecs, queries, k, list_names, title, markers, colors=None):
    list_of_list_of_vectors = []
    for i in queries:
        list_of_list_of_vectors.append(get_top_k(vecs, i, k))

    return test_sketchers(list_sketcher_classes, list_sizes, trials, list_of_list_of_vectors, queries, list_names, title, markers, colors)

def test_sketchers_asym(list_sketcher_classes, list_sizes, trials, list_of_list_of_vecs, queries, list_names, title):
    errors = []
    for sketcher_class in list_sketcher_classes:
        errors_class = []
        for size in list_sizes:
            error = 0
            for seed in range(0, trials):
                    sketcher = sketcher_class(size, seed)
                    matrix = sketcher.get_matrix()
                    for j in range(0, len(queries)):
                        if(type(queries[j]) == coo_array):
                            query = queries[j].toarray()
                        else:
                            query = queries[j]

                        vecs = list_of_list_of_vecs[j]
                        for vec in vecs:
                            vec_sketch = sketcher.sketch_asym(vec)
                            error += np.abs(vec_sketch.inner_product_asym(query, matrix) - vec @ query) / (np.linalg.norm(vec) * np.linalg.norm(query))
            errors_class.append(error / (len(queries) *  len(vecs) * trials))
        
        errors.append(errors_class)
                    

    colors = plt.cm.viridis(np.linspace(0, 1, len(errors)))

    for i in range(0, len(errors)):
        plt.plot(list_sizes, errors[i], marker='x', color=colors[i], label=list_names[i])


    plt.title(title)
    plt.xlabel("Sketch size")
    plt.ylabel("Error")
    #plt.ylim(0, 0.1)
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()
    return errors

def test_top_k_sketches_asym(list_sketcher_classes, list_sizes, trials, vecs, queries, k, list_names, title):
    list_of_list_of_vectors = []
    for i in queries:
        list_of_list_of_vectors.append(get_top_k(vecs, i, k))

    return test_sketchers_asym(list_sketcher_classes, list_sizes, trials, list_of_list_of_vectors, queries, list_names, title)

def test_pair_split(vec1, vec2, trials, sketcher_class, skethcer_class_tail, bits, split_function, title):
    splits = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    errors = []
    for split in splits:
        error = 0    
        for seed in range(0, trials):
            sketcher = sketcher_class(int(bits * split), int(bits * (1 - split)), seed, skethcer_class_tail, dim=len(vec1))

            error += np.abs((sketcher.sketch(vec1).inner_product_asym(vec2) - vec1 @ vec2)) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        errors.append(error / trials)


    adaptive_selected_entries = split_function(vec1, bits)
    adaptive_size = len(adaptive_selected_entries) * (np.ceil(np.log2(len(vec1))) + 16)

    plt.axvline(x=adaptive_size / bits, color='red')

    plt.plot(splits, errors, marker='x', color="green")

    plt.title(title)
    plt.xlabel("Split")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()
    return errors