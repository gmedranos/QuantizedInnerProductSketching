# Quantized Inner Product Sketching

## Setup and Prerequisites

Before running the code, you need to set up your environment and download the required datasets.

### 1. Environment & Directories
* **SPLADE:** Clone the [naver/splade](https://github.com/naver/splade) repository into a directory named `SPLADE` within your project root and install all of its dependencies.
* **Directories:** Create the following empty folders in your project root:
    * `figs/` (for storing generated figures)
    * `data/` (for storing data)

### 2. Datasets
You will need to download two datasets and place their files directly in the root of your project folder.

**MS MARCO Dataset:**
* Download the Passage Retrieval collection and queries from the [MS MARCO website](https://microsoft.github.io/msmarco/).
* Rename the extracted files to `collection.tsv` and `queries.dev.tsv` and place them in the project folder.

**arXiv Dataset:**
* Download the `instructorxl-arxiv-768.hdf5` file from this [Google Drive link](https://drive.google.com/drive/folders/1f76UCrU52N2wToGMFg9ir1MY8ZocrN34).
* Place the `.hdf5` file in the project folder.

---

## Running the Code

### Creating Embeddings
To generate the necessary embeddings, first, set the number of vectors you want to sketch by changing the parameter in line 41 of the super_script.py to your desired number.
Then, run the following command in your terminal:

```bash
python3 super_script.py -vectors
```

### Workshop Experiments
The experiments associated with the workshop paper are provided in Jupyter Notebooks. You can explore and run them by opening the following files:
* `workshop_arxiv.ipynb`
* `workshop_splade.ipynb`
* `workshop_synthetic_experiments.ipynb`
