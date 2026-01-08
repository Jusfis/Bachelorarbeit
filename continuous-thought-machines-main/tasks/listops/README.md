#ListOps Dataset

# Training
To train run this script from the root level of the repository:
```
bash tasks/listops/scripts/train_ctm.sh
```


# Analysis
Run analysis like in the other tasks:
```
python -m tasks.listops.analysis.run --log_dir <PATH_TO_LOG_DIR>
```


Download ListOps Dataset
# From Huggingface below:
```
mkdir raw_datasets
```

# Clone and unpack the LRA object.
# This can take a long time, so get comfortable.

```
rm -rf ./raw_datasets/lra_release.gz ./raw_datasets/lra_release  # Clean out any old datasets.
wget -v https://storage.googleapis.com/long-range-arena/lra_release.gz -P ./raw_datasets
```

# Add a progress bar because this can be slow.
```
pv ./raw_datasets/lra_release.gz | tar -zx -C ./raw_datasets/
```
