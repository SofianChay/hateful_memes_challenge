import torch
from torch import nn, optim
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
import copy
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import argparse

from model import MyVisualBert
from dataset import create

from time import gmtime, strftime
import os 
import jsonlines

from run import train

import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_splits', default=5, type=int)
    parser.add_argument('--dev_batch_size', default=32, type=int)
    parser.add_argument('--train_batch_size', default=16, type=int)
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--epochs', default=5, type=int)

    args = parser.parse_args()

    # create data
    train_set = []
    with jsonlines.open('data/train.jsonl', 'r') as f:
      for line in f: 
        train_set.append(line)

    dev_set = []
    with jsonlines.open('data/dev.jsonl', 'r') as f:
      for line in f: 
        dev_set.append(line)

    data = train_set + dev_set

    tmp = np.arange(0, len(data))
    np.random.shuffle(tmp)

    aucs = []

    for split in range(args.num_splits):
        print(f'split {split}')
        val_indexes = tmp[split * len(data) // args.num_splits: min((split + 1) * len(data) // args.num_splits, len(data))]
        train_indexes = [i for i in tmp if i not in val_indexes]
        train_set = [data[i] for i in train_indexes]
        dev_set = [data[i] for i in val_indexes]
        print('building dataloaders ...')
        train_dataloader = create(data=train_set, datatype='train', batch_size=args.train_batch_size)
        dev_dataloader = create(data=dev_set, datatype='dev', batch_size=args.dev_batch_size)
        print('done !')
        model = MyVisualBert()
        best_model, auc = train(train_dataloader, dev_dataloader, model, args.lr, args.epochs)
        aucs.append(auc)
        torch.save(best_model, f'saved_models/cross_val_{split}.pt')

    print(f'mean auc : {np.mean(aucs)}')


        



