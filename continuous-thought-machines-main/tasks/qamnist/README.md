# Q&A MNIST

## Training
To run the Q&A MNIST training that we used for the paper, run bash scripts from ~/continuous-thought-machines-main/ level of the repository:

Adjust the script for the model you want to train:
IMPORTANT! WANDB config may overwrite script parameters, so make sure to check the config in the wandb dashboard if you are using it.

```
bash tasks/qamnist/scripts/qamnist_{}.sh
```


To plot the results: 

run python script in plots and its subfolders.:
