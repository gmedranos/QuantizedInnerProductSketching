# Quantized Inner Product Sketching

### Requisites
To run this code you should first clone [this repo](https://github.com/naver/splade) into a folder named SPLADE and install all of it's dependencies, create one folder for the figures, called "figs" and a folder for the data, 
called "data". You should also download the dataset [MSMARCO](https://microsoft.github.io/msmarco/). I am using the Passage Retrieval collection and queries, that should be named "collection.tsv" and "queries.dev.tsv", 
and should be put in the project folder.

### Running the code

To run the code, one should use the command `python3 super_script.py -flags`, where flags is one or more of the following:

1. -vectors : create the embeddings for the vectors
2. -sketchesSH: create the SimHash sketches for the vectors that were created with the -vectors
3. -sketchesPSQ: create the Quantized Priority sketches for the vectors that were created with the -vectors
4. -q: create the embeddings for the queries vectors
5. -ip: test the inner product from random samples from vectors
6. -recall: test the recall

Given the size of the test, it might take a while to run, so you can change the values in line 32 of the super_script.py so you run it in a smaller dataset.
