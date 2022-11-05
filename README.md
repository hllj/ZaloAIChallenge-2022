# PytorchLightning - Hydra - Wandb

This is a powerful and efficient template for training and testing ML models.

## Installation

Install the requirements by following the command:

```bash
cd $this_project_directory$
conda env create -p $path_to_your_envs$ -f environment.yml
```

## Usage

PytorchLightning helps you handle all the residual things, such as assigning tensors to respective devices, creating training loop, running on ddp process if you pre-define numbers of GPUs.  

All you need to do in this template is **defining your model's architecture and training process**, **creating correct dataloaders** and **login to wandb** in order to visualize your results.
 
### Login to Wandb
After installing conda environment, run:

```bash
wandb login --relogin
```

to login using your own account.
### Define your model and training process
One best practice is creating a subfolder inside **src** for every model you want to train.

Basically, a model needs at least 3 correspond files: **dataset.py**, **model.py**, **networks.py**.

#### src/dataset.py ####
This file helps define your dataset and dataloader

#### src/model.py ####
Put your codes relating to your training and testing step here, including define your model, loss function, optimizer and so on.

#### src/networks.py ####
Put your network's architecture here.

#### main.py ####
The main file for running the process you have just defined above. We use hydra for importing config. Define your config path and config file name as kwargs of @hydra.main static function. Define your training process and validating process also. Define your visualization structure by editing the **LogPredictionSamplesCallback** function.

#### configs/$your_config_name.yaml$ ####
Define the variables you need for running your model, including your model's hyperparams, dataset's hyperparams, optimizations, etc ... 

## Running ##

```bash
CUDA_VISIBLE_DEVICES=$list_of_gpus_id$ python main.py
```