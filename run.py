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

def training_step(optimizer, train_dataloader, model, criterion, scheduler, device):
    loss = 0
    model.train()
    tk = tqdm(train_dataloader, total=len(train_dataloader), unit="batch")
    total_examples = 0
    for i, batch in enumerate(tk):
        total_examples += len(batch[0])
        optimizer.zero_grad()
        logits = model(input_ids=batch[0].to(device), attention_mask=batch[1].to(device), visual_embeddings=batch[3].to(device))
        batch_loss = criterion(logits, batch[2].to(device))
        loss += batch_loss.item() 
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        tk.set_postfix({"average loss": loss / total_examples})
    return loss 


def test(model, dev_dataloader):
    print("validation ...")
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    model.eval()
    num_examples = 0
    confidences = []
    labels = []
    accuracy = 0
    with torch.set_grad_enabled(False):
        
        for i, batch in enumerate(tqdm(dev_dataloader, unit="batch")): 
            num_examples += len(batch[0])
            logits = model(input_ids=batch[0].to(device), attention_mask=batch[1].to(device), visual_embeddings=batch[3].to(device))
            probs = F.softmax(logits, dim=1)
            _, predicted_classes = probs.max(dim=1)
            accuracy += torch.sum(predicted_classes.cpu() == batch[2]).item()
            confidences += probs[:, 1].cpu().tolist()
            labels += batch[2].tolist()
    
    roc_auc = roc_auc_score(labels, confidences)
    return roc_auc, accuracy / num_examples


def train(train_dataloader, dev_dataloader, model, lr, num_epochs):
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    best_model = copy.deepcopy(model.state_dict())
    max_auc = 0

    print("training start ! ")
    print("fine tuning ... ")
    optimizer = optim.AdamW(model.parameters(), lr=lr, eps=1e-8)
    total_steps = num_epochs * len(train_dataloader) 
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        print(f"fine tuning step : {epoch + 1}")
        loss = training_step(optimizer, train_dataloader, model, criterion, scheduler, device)
        print(f"training loss : {loss}")
        auc_roc, accuracy = test(model, dev_dataloader)
        print(f'accuracy : {accuracy}, auc roc : {auc_roc}')
        if auc_roc > max_auc:
          max_auc = auc_roc
          best_model = copy.deepcopy(model.state_dict()) 

    return best_model, max_auc


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--epochs', default=6, type=int)
    parser.add_argument('--train_batch_size', default=16, type=int)
    parser.add_argument('--dev_batch_size', default=32, type=int)

    args = parser.parse_args()

    setattr(args, 'model_time', strftime('%H:%M:%S', gmtime()))

    model = MyVisualBert()
    # model.load_state_dict(torch.load('saved_models/23:19:15.pt'))
    train_set = []
    with jsonlines.open('data/train.jsonl', 'r') as f:
      for line in f: 
        train_set.append(line)

    dev_set = []
    with jsonlines.open('data/dev.jsonl', 'r') as f:
      for line in f: 
        dev_set.append(line)

    print('building dataloaders ...')
    train_dataloader = create(data=train_set, datatype='train', batch_size=args.train_batch_size)
    dev_dataloader = create(data=dev_set, datatype='dev', batch_size=args.dev_batch_size)
    print('done !')
    # print(test(model, dev_dataloader))

    best_model = train(train_dataloader, dev_dataloader, model, args.lr, args.epochs)

    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    
    torch.save(best_model, f'saved_models/{args.model_time}.pt')
    print('model saved !')
