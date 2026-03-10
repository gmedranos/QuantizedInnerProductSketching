from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import pickle
import time
import datetime

def create_matryoshka(num_vectors):
    t = time.time()

    # Load a Matryoshka-capable model
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device="cpu")
    # Generate full-sized embeddings (e.g., 768 dimensions)

    df = pd.read_csv("./collection.tsv", on_bad_lines='warn', sep='\t', header=None)
    docs_encoded = []

    for i in range(0, num_vectors):
        print(i)
        # now compute the document representation
        doc = df[df.columns[1]][i]
        doc_encoding = model.encode(doc)
        docs_encoded.append(doc_encoding)

    with open('data/matryoshka.pkl', 'wb') as f:
        pickle.dump(docs_encoded, f)

    t_e = time.time()
    print("Time to matryoshka: ")
    print(datetime.timedelta(seconds = t_e - t))

# Create a file containing the splade embedding of num_vectors entries from MSMARCO
def create_queries(num_vectors):
    t = time.time()

    # Load a Matryoshka-capable model
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device="cpu")
    # Generate full-sized embeddings (e.g., 768 dimensions)

    df = pd.read_csv("./queries.dev.tsv", on_bad_lines='warn', sep='\t', header=None)
    docs_encoded = []

    for i in range(0, num_vectors):
        print(i)
        # now compute the document representation
        doc = df[df.columns[1]][i]
        doc_encoding = model.encode(doc)
        docs_encoded.append(doc_encoding)

    with open('data/matryoshka_q.pkl', 'wb') as f:
        pickle.dump(docs_encoded, f)

    t_e = time.time()
    print("Time to matryoshka: ")
    print(datetime.timedelta(seconds = t_e - t))

create_queries(10)