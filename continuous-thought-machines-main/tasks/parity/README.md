# Parity

## Training
To run the parity training that we used for the paper, run bash scripts from the ~/continuous-thought-machines-main/ level of the repository:
Adjust the script for the model you want to train:
IMPORTANT! WANDB config may overwrite script parameters, so make sure to check the config in the wandb dashboard if you are using it.


```
bash tasks/parity/scripts/parity_{}.sh
```

To plot the results: 

run python script in plots and its subfolders.:
