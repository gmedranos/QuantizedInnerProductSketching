import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModel
import sys
import os
import numpy as np
import pickle
import pandas as pd
cwd = os.getcwd()
sys.path.append(cwd)

from SPLADE.splade.models.transformer_rep import Splade


# Create a file containing the splade embedding of num_vectors entries from MSMARCO
def create_vector(num_vectors):
    model_type_or_dir = "naver/splade-cocondenser-ensembledistil"

    model = Splade(model_type_or_dir, agg="max")

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
    reverse_voc = {v: k for k, v in tokenizer.vocab.items()}


    df = pd.read_csv("./collection.tsv", on_bad_lines='warn', sep='\t', header=None)
    vectors = []

    for i in range(0, num_vectors):
        # now compute the document representation
        doc = df[df.columns[1]][i]
        with torch.no_grad():
            doc_rep = model(d_kwargs=tokenizer(doc, return_tensors="pt"))["d_rep"].squeeze()  # (sparse) doc rep in voc space, shape (30522,)
        vectors.append((i, np.array(doc_rep)))


    with open('data/arrays.pkl', 'wb') as f:
        pickle.dump(vectors, f)
