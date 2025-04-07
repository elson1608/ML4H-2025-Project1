## Dependencies
The necessary python libraries are listed in `requirements.txt` and can be installed via
```
pip install -r requirements.txt
```
Also `ollama` needs to be installed to recreate the experiments for the foundation models (https://ollama.com/download). 
\
\
The original data can be found at https://physionet.org/content/challenge-2012/1.0.0/
\
\
The experiments were conducted in `Python 3.11`
## Data generation
### Data Processing and Exploration 
The initial data-transformation can be found in `code/data-processing-and-exploration/Q1.1-data-transformation.ipynb` and will generate the 
`set-{a,b,c}.parquet` files.

**Important disclaimer:**
We slightly deviated from the handout here and kept the ICUType column, because we used in for the exploratory data analysis.

The preprocessing of the transformed data is done in `code/data-processing-and-exploration/Q1.3-preprocess-data-for-machine-learning.ipynb` and will generate the 
`set-{a,b,c}-imputed.parquet` as well as the `set-{a,b,c}-imputed-scaled.parquet` files.

### Supervised Learning
The feature extraction with `tsfresh` was done in script `code/supervised-learning/feature_extraction.py` ans will generate the `tsfresh-set-{a,b,c}.parquet` files
\
\
The tokenized time series representation was generated using the script `code/supervised-learning/generate_triplets.py` and will generate the `set-{a,b,c}-triplet.parquet` files.  

### Representation Learning
Here we did not need to generate new data

### Foundation Models
The script `llm-solving.py` exports the predicted probabilties of `gemma2:2b` in the `foundation-model-predictions.parquet` file.
\
\
The script `llm-embeddings.py` was used to export the embeddings of `gemma2:2b` in the `set-{a,c}-foundation-model-embeddings.parquet` files.
\
\
The script `chronos-embeddings.py` stores the embedding tensors of `chronos` in the `set-{a,b,c}-chronos-embeddings.pt` files.  

## Project Structure
The `code` folder contains a folder for each of the four tasks containing the jupyter notebooks that can be run after data generation to reproduce our experiments.
\
\
We stored the data in a seperate folder called `data` which is not part of this repository/submission 