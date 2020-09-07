import argparse
import pickle

import jsonlines
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch import mode

from dataset import dataset
from utils.seed import set_seed
import torch
from model.visualbert import MyVisualBert, VisualBertWithPooler
from torch.cuda import amp
from transformers import AdamW, get_linear_schedule_with_warmup
from model import engine
import numpy as np


def run(lr, epochs, num_models, train_bs, valid_bs, image_features, loss, seed):
    # Load data
    train_set = []
    with jsonlines.open('data/train.jsonl', 'r') as f:
        for line in f:
            train_set.append(line)

    dev_set = []
    with jsonlines.open('data/dev.jsonl', 'r') as f:
        for line in f:
            dev_set.append(line)

    print('building dataloaders ...')
    with open(f'{image_features}/images_features_dict.pkl', 'rb') as f:
        images_features_dict = pickle.load(f)
    df_train = pd.DataFrame(train_set)
    df_dev = pd.DataFrame(dev_set)

    print("train set")
    train_dataloader = dataset.create(data=df_train.to_dict('records'), datatype='train',
                                      batch_size=train_bs, images_features_dict=images_features_dict)
    print("dev set")
    valid_dataloader = dataset.create(data=df_dev.to_dict('records'), datatype='valid',
                                      batch_size=valid_bs, images_features_dict=images_features_dict)
    print('done !')

    scores = []

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
    print(f'Working on {device}')

    for i in range(num_models):
        print(f'Processing model {i}')
        seed = np.random.randint(low=42, high=2020)
        set_seed(seed)

        # Model
        model = VisualBertWithPooler()
        # model = MyVisualBert()
        model.to(device)

        # AMP Scaler (https://arxiv.org/pdf/1710.03740.pdf)
        scaler = amp.GradScaler()

        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                'params': [
                    p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay':
                0.001
            },
            {
                'params': [
                    p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay':
                0.0
            },
        ]

        num_train_steps = int(len(df_train) / train_bs * epochs)
        optimizer = AdamW(optimizer_parameters, lr=lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

        model_path = f'saved_models/model_{i}.bin'

        best_auc = 0

        for epoch in range(epochs):
            engine.train(train_dataloader, model, optimizer,
                         scheduler, device, scaler)
            valid_roc_auc = engine.evaluation(valid_dataloader, model, device)
            print(f'Epoch: {epoch}, validaion ROC AUC: {valid_roc_auc}')
            if valid_roc_auc > best_auc:
                best_auc = valid_roc_auc
                torch.save(model.state_dict(), model_path)
        scores.append(best_auc)
    print(f'Ensemble auc mean: {np.mean(scores)}')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', default=5e-3, type=float)
    parser.add_argument('--epochs', default=13, type=int)
    parser.add_argument('--train_batch_size', default=32, type=int)
    parser.add_argument('--valid_batch_size', default=32, type=int)
    parser.add_argument('--num_models', default=12, type=int)
    parser.add_argument('--image_features', default='mask_rcnn_features')
    parser.add_argument('--loss', default='rocstar',
                        help='"margin", "focal", "bce", "rocstar"')
    parser.add_argument('--seed', default=42)

    args = parser.parse_args()

    run(lr=args.lr, epochs=args.epochs, num_models=args.num_models, train_bs=args.train_batch_size,
        valid_bs=args.valid_batch_size, image_features=args.image_features, loss=args.loss, seed=args.seed)
