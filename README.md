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

#### src/dataset.py

This file helps define your dataset and dataloader.

#### src/model.py

Put your codes relating to your training and testing step here, including define your model, loss function, optimizer and so on.

#### src/networks.py

Put your network's architecture here.

#### main.py

The main file for running the process you have just defined above. We use hydra for importing config. Define your config path and config file name as kwargs of @hydra.main static function. Define your training process and validating process also. Define your visualization structure by editing the **LogPredictionSamplesCallback** function.

#### configs/$your_config_name.yaml$

Define the variables you need for running your model, including your model's hyperparams, dataset's hyperparams, optimizations, etc ...

#### Add Zalo AI Challenge Dataset

Download train.zip and public.zip, place in data/

Run command to extract frames and split data. Then choose option 1 or 2 to create type of data (no padding or padding)

#### Option 1: No padding, keep resolution

```bash
cd data/
unzip train.zip
unzip public.zip
unzip public_test_2.zip
python get_frame.py -i train/videos/ -o train/images/
python get_frame.py -i public/videos/ -o public/images/
python get_frame.py -i public_test_2/videos/ -o public_test_2/images/
python create_data.py -dir train/ -images images_png -l label1.csv
```

#### Option 2: Add padding, ratio 1:1

```bash
cd data/
unzip train.zip
unzip public.zip
unzip public_test_2.zip
python get_frame.py -i train/videos/ -o train/padding_images/ -p
python get_frame.py -i public/videos/ -o public/padding_images/ -p
python get_frame.py -i public_test_2/videos/ -o public_test_2/padding_images/ -p
python create_data.py -dir train/ -images padding_images -l label.csv
```

train_list: data/train/train.csv
val_list: data/train/val.csv

## Running

**Run**

```bash
CUDA_VISIBLE_DEVICES=$list_of_gpus_id$ python main.py
```


## Docker

```bash
docker run --gpus '"device=0"' --network host -it --name zac2022 pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime  /bin/bash

apt update
apt install software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt install python3.9
apt install python3.9-distutils
python3.9 -m pip install --upgrade pip
python3.9 -m pip install --upgrade setuptools

python3.9 -m pip install -r pip_env.txt

docker run -it --gpus '"device=0"' --network host -v /storage/cv_hcm/zaloai2022/private_test/videos:/data -v /storage/cv_hcm/zaloai2022/result_private:/result zac2022:v2 /bin/bash  /code/predict.sh

```