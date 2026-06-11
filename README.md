# Quantized Inner Product Sketching

### Requisites
To run this code you should first clone [this repo](https://github.com/naver/splade) into a folder named SPLADE and install all of it's dependencies, create one folder for the figures, called "figs" and a folder for the data, 
called "data". You should also download the dataset [MSMARCO](https://microsoft.github.io/msmarco/). I am using the Passage Retrieval collection and queries, that should be named "collection.tsv" and "queries.dev.tsv", 
and should be put in the project folder.

### Running the code

To create the embeddings, one should use the command `python3 super_script.py -vectors`

The experiments for the workshop paper are available in the three notebooks: "workshop_arxiv.ipynb", "workshop_splade.ipynb" and "workshop_synthetic_experiments.ipynb"
