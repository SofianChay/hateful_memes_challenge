import torch
from torch import nn, optim
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
import copy
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import argparse
import numpy as np 

from utils.model import MyVisualBert, RobertaClassif
# from bertweet.model.model import VisualBerTweet
from utils.dataset import create

from time import gmtime, strftime
import os 
import jsonlines
import pickle 

from utils.losses import FocalLoss, RocStar, MarginFocalLoss

from transformers import RobertaConfig, RobertaTokenizer, BertTokenizer, BertConfig

os.environ['OC_DISABLE_DOT_ACCESS_WARNING'] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
map_losses = {"bce": nn.CrossEntropyLoss, "focal": FocalLoss, 'rocstar': RocStar, "margin": MarginFocalLoss}
map_models = {"visualbert": MyVisualBert, "roberta": RobertaClassif}


def training_step(epoch, optimizer_hate, train_dataloader, model, criterion, scheduler_hate, device, name_loss, last_epoch_y_pred=None, last_epoch_y_t=None):
    loss_hate = 0
    model.train()
    tk = tqdm(total=len(train_dataloader), unit="batch")
    total_examples_hate = 0

    if epoch == 0 and name_loss == 'rocstar':
        last_epoch_y_pred = torch.tensor( 1.0 - np.random.rand(len(train_dataloader))/2.0, dtype=torch.float, device=device)
        last_epoch_y_t    = torch.tensor(np.random.randint(0, 2, len(train_dataloader)),dtype=torch.float, device=device)
    if name_loss == 'rocstar':
        epoch_y_pred = []
        epoch_y_t = []
    else:
        epoch_y_pred, epoch_y_t = None, None

    i = 0
    iter_train = iter(train_dataloader)
    while i < len(train_dataloader):
        optimizer_hate.zero_grad()
        batch = next(iter_train)
        total_examples_hate += len(batch[0])
        if type(batch[3]) == torch.Tensor:
            logits = model(input_ids=batch[0].to(device), attention_mask=batch[1].to(device), visual_embeddings=batch[3].to(device))
        else:
            logits = model(input_ids=batch[0].to(device), attention_mask=batch[1].to(device), visual_embeddings=None)
        if name_loss == 'rocstar':
            preds = torch.sigmoid(logits)
            batch_loss = criterion(_y_true=batch[2].to(device), y_pred=preds, _epoch_true=last_epoch_y_t.to(device), epoch_pred=last_epoch_y_pred.to(device))
        else:
            batch_loss = criterion(logits, batch[2].to(device))
        loss_hate += batch_loss.item() 
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_hate.step()
        scheduler_hate.step()
        i += 1
        tk.update(1)
        tk.set_postfix({"average loss hate": loss_hate / total_examples_hate})
        if name_loss == 'rocstar':
            epoch_y_pred.extend(preds.cpu().tolist())
            epoch_y_t.extend(batch[2].to(device).tolist())
    if name_loss == 'rocstar':
        epoch_y_t = torch.tensor(epoch_y_t)
        epoch_y_pred = torch.tensor(epoch_y_pred)
        criterion.epoch_update_gamma(epoch_y_t, epoch_y_pred)
    return loss_hate, epoch_y_pred, epoch_y_t


def test(model, dev_dataloader, args, epoch):
    print("validation ...")
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    num_examples = 0
    confidences = []
    labels = []
    accuracy = 0
    if args.identify_fp and epoch == args.epochs - 1:
        fp_ids = []
    with torch.set_grad_enabled(False):
        
        for i, batch in enumerate(tqdm(dev_dataloader, unit="batch")): 
            num_examples += len(batch[0])
            if type(batch[3]) == torch.Tensor:
                logits = model(input_ids=batch[0].to(device), attention_mask=batch[1].to(device), visual_embeddings=batch[3].to(device))
            else:
                logits = model(input_ids=batch[0].to(device), attention_mask=batch[1].to(device), visual_embeddings=None)
            probs = torch.sigmoid(logits)
            # probs = F.softmax(logits, dim=1)
            # _, predicted_classes = probs.max(dim=1)
            predicted_classes = (probs >= .5)
            accuracy += torch.sum(predicted_classes.cpu() == batch[2]).item()

            # confidences += probs[:, 1].cpu().tolist()
            confidences += probs.cpu().tolist()
            labels += batch[2].tolist()
            if args.identify_fp and epoch == args.epochs - 1:
              for j in range(len(batch[4])):
                  if predicted_classes[j] != batch[2][j]:
                      fp_ids.append((batch[4][j].item(), batch[2][j].item()))
              
    if args.identify_fp and epoch == args.epochs - 1:
        with open('fp_ids.txt', 'w') as f:
            for id_ in fp_ids:
                print(id_, file=f) 

    
    roc_auc = roc_auc_score(labels, confidences)
    return roc_auc, accuracy / num_examples


def train(train_dataloader, dev_dataloader, model, lr, num_epochs, name_loss, args):
    best_model = copy.deepcopy(model.state_dict())
    max_auc = 0

    print("training start ! ")
    print("fine tuning ... ")
    optimizer_hate = optim.AdamW(model.parameters(), lr=lr, eps=1e-8)
    total_steps = num_epochs * len(train_dataloader) 
    scheduler_hate = get_linear_schedule_with_warmup(optimizer_hate, num_warmup_steps=0, num_training_steps=total_steps)
    criterion = map_losses[name_loss]()

    for epoch in range(num_epochs):
        if epoch == 0:
          epoch_y_pred, epoch_y_t = None, None 
        print(f"fine tuning step : {epoch + 1}")
        loss, epoch_y_pred, epoch_y_t = training_step(epoch, optimizer_hate, train_dataloader, model, criterion, scheduler_hate, device, name_loss, epoch_y_pred, epoch_y_t)
        print(f"training loss : {loss}")
        auc_roc, accuracy = test(model, dev_dataloader, args, epoch)
        print(f'accuracy : {accuracy}, auc roc : {auc_roc}')
        if auc_roc > max_auc:
          max_auc = auc_roc
          best_model = copy.deepcopy(model.state_dict()) 

    return best_model, max_auc


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--epochs', default=6, type=int)
    parser.add_argument('--train_batch_size', default=16, type=int)
    parser.add_argument('--dev_batch_size', default=32, type=int)
    parser.add_argument('--num_models', default=12, type=int)
    parser.add_argument('--images_features', default='mask_rcnn_features')
    parser.add_argument('--loss', default='margin', help='"margin", "focal", "bce", "rocstar"')
    parser.add_argument('--identify-fp', default=0, type=int)
    parser.add_argument('--model', default='visualbert')

    args = parser.parse_args()

    args.identify_fp = args.identify_fp == 1

    print(f"training {args.model}")

    train_set = []
    with jsonlines.open('data/train.jsonl', 'r') as f:
      for line in f: 
        train_set.append(line)


    dev_set = []
    with jsonlines.open('data/dev.jsonl', 'r') as f:
      for line in f: 
        dev_set.append(line)


    print('building dataloaders ...')
    if args.model == 'visualbert':
        with open(f'{args.images_features}/images_features_dict.pkl', 'rb') as f:
          images_features_dict = pickle.load(f)
    else:
        images_features_dict = None
    

    if args.model == 'visualbert':
        config = BertConfig.from_pretrained('bert-base-uncased')
        tkz = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        config = RobertaConfig.from_pretrained('roberta-base')
        tkz = RobertaTokenizer.from_pretrained('roberta-base')

    print("train set")
    train_dataloader = create(data=train_set, datatype='train', batch_size=args.train_batch_size, images_features_dict=images_features_dict, tkz=tkz, config=config)
    print("dev set")
    dev_dataloader = create(data=dev_set, datatype='dev', batch_size=args.dev_batch_size, images_features_dict=images_features_dict, tkz=tkz, config=config)
    print('done !')
    
    del images_features_dict
    
    scores = []
    for i in range(args.num_models):
        print(f"model {i}")
        model = map_models[args.model]().to(device)
        best_model, auc = train(train_dataloader, dev_dataloader, model, args.lr, args.epochs, args.loss, args)
        scores.append(auc)
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
        
        torch.save(best_model, f'saved_models/{args.model}_ensemble_{i}.pt')
        print('model saved !')

    print(f"mean auc : {np.mean(scores)}")
