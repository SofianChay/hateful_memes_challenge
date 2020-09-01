from tqdm import tqdm
from utils.averagemeter import AverageMeter
from utils.losses import MarginFocalLoss
import torch
from torch.cuda import amp
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score


def train(data_loader, model, optimizer, scheduler, device, scaler):
    model.train()
    losses = AverageMeter()
    tqdm_data = tqdm(data_loader, total=len(data_loader))
    for data in tqdm_data:
        # Fetch data
        input_ids = data[0].to(device)
        attention_mask = data[1].to(device)
        targets = data[2].to(device)
        visual_embeddings = data[3].to(device)
        # Compute model output
        optimizer.zero_grad()
        with amp.autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask,
                           visual_embeddings=visual_embeddings)
            loss = MarginFocalLoss()(
                logits, targets)
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.update(loss.item(), input_ids.size(0))
        scheduler.step()
        tqdm_data.set_postfix(loss=losses.avg)
    torch.cuda.empty_cache()


def evaluation(data_loader, model, device):
    model.eval()
    prediction_list = []
    target_list = []
    with torch.no_grad():
        for data in data_loader:
            # Fetch data
            input_ids = data[0].to(device)
            attention_mask = data[1].to(device)
            targets = data[2].to(device)
            visual_embeddings = data[3].to(device)

            # Compute model output
            with amp.autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask,
                               visual_embeddings=visual_embeddings)
            preds = torch.sigmoid(logits).cpu(
            ).detach().numpy().reshape(-1, 1)
            prediction_list.append(preds)
            target_list.append(targets.cpu().detach().numpy())
    torch.cuda.empty_cache()

    roc_auc = roc_auc_score(np.concatenate(target_list),
                            np.concatenate(prediction_list))
    return roc_auc


def predict(data_loader, model, device):
    model.eval()
    prediction_list = []
    ids_list = []
    tqdm_data = tqdm(data_loader, total=len(data_loader))
    with torch.no_grad():
        for data in tqdm_data:
            # Fetch data
            input_ids = data[0].to(device)
            attention_mask = data[1].to(device)
            visual_embeddings = data[3].to(device)
            ids = data[4]
            # Compute model output
            logits = model(input_ids=input_ids, attention_mask=attention_mask,
                           visual_embeddings=visual_embeddings)
            preds = torch.sigmoid(logits).cpu(
            ).detach().numpy().reshape(-1, 1)
            prediction_list.append(preds)
            ids_list.append(ids.cpu().detach().numpy())
    torch.cuda.empty_cache()
    ids = np.concatenate(ids_list)
    predictions = np.concatenate(prediction_list)

    return ids, predictions
