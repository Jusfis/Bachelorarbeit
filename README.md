# Bachelorarbeit
Bachelor Thesis on the Topic on Kolmogorov Arnold Networks for Continuous Thought Machines 

This Repository consists of an aggregation of useful repositories and contents for the thesis.

Project Structure from  └── continuous-thought-machines-main on
├── data
│   ├── ListOps
│   ├── MNIST   
│   └── mazes
├── dataset
├── logs
│    ├── listops   
│    ├── mazes
│    ├── parity
│    └── qamnist
├── models
├── parity_logs_kan
├── tasks
│   ├── image_classification
│   │   ├── analysis
│   │   └── scripts
│   ├── listops
│   │   ├── analysis
│   │   ├── dataset
│   │   ├── depr
│   │   └── scripts
│   ├── mazes
│   │   ├── analysis
│   │   └── scripts
│   ├── other
│   │   ├── rl
│   │   └── sort
│   ├── parity
│   │   ├── analysis
│   │   ├── depr
│   │   ├── logs
│   │   ├── model
│   │   ├── scripts
│   │   └── wandb
│   └── qamnist
│       ├── analysis
│       └── scripts
├── tests
└── utils
/////
.
├── data/               # Raw datasets (MNIST, ListOps, Mazes)
├── dataset/            # Processed data and loaders
├── logs/               # Training logs & checkpoints (organized by task)
├── models/             # Core architecture definitions
├── tasks/              # Task-specific logic & experiments
│   ├── listops/        # Scripts, analysis, and task-specific datasets
│   ├── mazes/          # Pathfinding experiments
│   ├── parity/         # Parity & logic tasks (incl. wandb logs)
│   └── ...             # Other modules (image_classification, qamnist)
├── tests/              # Unit tests for core components
└── utils/              # Helper functions & shared tools

to install efficient-kan package:
```bash
pip install git+https://github.com/Blealtan/efficient-kan.git
```

# ListOps Facebook/Meta download link
[Listops Meta link](https://github.com/facebookresearch/latent-treelstm/blob/master/data/listops/external/urls.txt)

# ListOps Google download link
[Listops Google link](https://github.com/google-research/long-range-arena)




