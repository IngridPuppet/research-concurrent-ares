import argparse as ap
import logging
import os
import pathlib
import sys

import atom3d.datasets as da
import dotenv as de
import pytorch_lightning as pl
import pytorch_lightning.loggers as log
import torch_geometric
import wandb

import ares.data as d
import ares.model as m


root_dir = pathlib.Path(__file__).parent.parent.absolute()
de.load_dotenv(os.path.join(root_dir, '.env'))
logger = logging.getLogger("lightning")
wandb.init(project="ares")


def main():
    parser = ap.ArgumentParser()
    # add PROGRAM level args
    parser.add_argument('train_dataset', type=str)
    parser.add_argument('val_dataset', type=str)
    parser.add_argument('-f', '--filetype', type=str, default='lmdb',
                        choices=['lmdb', 'pdb', 'silent'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--label_dir', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=1)

    # add model specific args
    parser = m.ARESModel.add_model_specific_args(parser)

    # add trainer args
    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    dict_args = vars(hparams)

    transform = d.create_transform(True, hparams.label_dir, hparams.filetype)

    # DATA PREP
    logger.info(f"Dataset of type {hparams.filetype}")

    logger.info(f"Creating dataloaders...")
    train_dataset = da.load_dataset(hparams.train_dataset, hparams.filetype,
                                    transform=transform)
    train_dataloader = torch_geometric.data.DataLoader(
        train_dataset,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        shuffle=True)
    val_dataset = da.load_dataset(hparams.val_dataset, hparams.filetype,
                                  transform=transform)
    val_dataloader = torch_geometric.data.DataLoader(
        val_dataset,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers)

    tfnn = m.ARESModel(**dict_args)
    wandb_logger = log.WandbLogger(save_dir=os.environ['MODEL_DIR'])
    trainer = pl.Trainer.from_argparse_args(hparams, logger=wandb_logger)

    # TRAINING
    logger.info("Running training...")
    out = trainer.fit(tfnn, train_dataloader, val_dataloader)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(process)d: ' +
                        '%(message)s',
                        level=logging.INFO)
    main()
