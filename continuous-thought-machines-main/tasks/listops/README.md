# ListOps Dataset

# Training
To train run this script from ~/continuous-thought-machines-main/ level of the repository:
Adjust the script for the model you want to train:
IMPORTANT! WANDB config may overwrite script parameters, so make sure to check the config in the wandb dashboard if you are using it.

```
bash tasks/listops/scripts/listops_{}.sh
```


To generate train and test data, run the make_data_nyul-ml.py script:

```
python -m tasks.listops.make_data_nyul_ml 

```
To plot the results: 
run python script in plots folder:
