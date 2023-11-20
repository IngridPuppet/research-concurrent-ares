# Code for ARES neural network.

## Overview

Training and inference code for ARES network from "Geometric Deep Learning of RNA Structure."

## Installation

### Download e3nn

ARES requires an adapted version of the e3nn package. To install it, please download it from https://doi.org/10.5281/zenodo.5090150 and place it in `lib` folder.

### Create conda environment

```
conda env create -f environment.yml
conda activate ares_release
```

Note, that depending on the version of cuda you have installed, you may need to update the `cudatoolkit` version in the `environment.yml` (it is currently set for CUDA 10.1).  You can also remove it entirely if you want to use cpu-only.

### Set environment variables

Copy the environment template to set your environment variables, and update your new `.env` as necessary (in particular the ROOT_DIR variable):

`cp .env.template .env`

## Usage

### Inference

To make a prediction, the general format is as follows:

`python -m ares.predict input_dir checkpoint.ckpt output.csv -f [pdb|silent|lmdb] [--nolabels] [--label_dir=score_dir]`

For example, to predict on the two test pdbs included in the repository, using the original ARES model:

`python -m ares.predict data/sample/pdbs data/weights.ckpt output.csv -f pdb --nolabels`

For a 100-nucleotide RNA, this should run at 5 s/it on a single CPU.  The expected output in `output.csv` for the above command would be (with possible fluctuation in up to 7th decimal place):

```
pred,id,rms,file_path
6.0303569,S_000028_476.pdb,0,/PATH/TO/ares_release/data/pdbs/S_000028_476.pdb
7.9565406,S_000041_026.pdb,0,/PATH/TO/ares_release/data/pdbs/S_000041_026.pdb
```

### Training

To use training data equivalent to that used to train the original ARES, you will need to download it from https://purl.stanford.edu/bn398fc4306, under classics_train_val.tar

To train the model on CPU, you can then run the following command:

`python -m ares.train /path/to/lmdbs/train /path/to/lmdbs/val -f lmdb --batch_size=8 --accumulate_grad_batches=2 --learning_rate=0.0005 --max_epochs=5 --num_workers=8`

Note this will run quite slowly.  To run faster, consider using a GPU (see below).

### Using a GPU

You can enable a gpu with the `--gpus` flag.  It is also recommended to provision additional CPUs with the `--num_workers` flags (more is better). GPU should have at least 12GB of memory.  For example to predict:

`python -m ares.predict data/sample/pdbs data/weights.ckpt output.csv -f pdb --nolabels --gpus=1 --num_workers=8`

For a 100-nucleotide RNA, this should run at 2 it/s on a Titan X Pascal GPU.  

And to train:

`python -m ares.train /path/to/lmdbs/train /path/to/lmdbs/val -f lmdb --batch_size=8 --accumulate_grad_batches=2 --learning_rate=0.0005 --max_epochs=5 --gpus=1 --num_workers=8`


### Create input LMDB dataset

The LMDB data format allows for fast, random access to your data, which is useful for machine learning workloads.  To convert a PDB dataset to the LMDB format, you can run:

`python -m atom3d.datasets input_dir output_lmdb -f pdb [--score_path=/path/to/scores.sc]`

For example, can use the sample data to create one:

`python -m atom3d.datasets data/sample/pdbs data/sample/lmdb -f pdb`
