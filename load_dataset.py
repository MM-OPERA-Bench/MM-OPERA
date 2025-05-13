from datasets import config
from datasets import load_dataset
config.HF_DATASETS_CACHE = './dataset'

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("titic/MM-OPERA")

# Example of an RIA instance
ria_example = ds["ria"][0]
print(ria_example)

# Example of an ICA instance
ica_example = ds["ica"][0]
print(ica_example)
