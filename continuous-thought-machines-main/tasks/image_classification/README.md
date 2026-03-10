# Image classification

This folder contains code for training and analysing imagenet and cifar related experiments. 

To run the Q&A MNIST training that we used for the paper, run bash scripts from the ~/continuous-thought-machines-main/ level of the repository:

Adjust the script for the model you want to train:
IMPORTANT! WANDB config may overwrite script parameters, so make sure to check the config in the wandb dashboard if you are using it.

# Training
```
bash tasks/qamnist/scripts/qamnist_{}.sh
```

To plot the results: 

run python script in plots and its subfolders.:




only if using imagenet: 
## Accessing and loading imagenet

To get this to work for you, you will need to do the following:
1. Login to huggingface (make an account) to agree to TCs of this dataset, 
2. Make a new access token.
3. Install huggingface_hub on the target machine with ```pip install huggingface_hub``` 
4. Run ```huggingface-cli login``` and use your token. This will authenticate you on the backend and allow the code to run.
5. Simply run an imagenet experiment. It will auto download and do all that magic. 
