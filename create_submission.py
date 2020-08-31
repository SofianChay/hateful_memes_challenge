import pandas as pd 
import torch
import torch.nn.functional as F
import os 
import argparse 
from utils.model import MyVisualBert
from tqdm import tqdm
from utils.dataset import create 
import numpy as np
import jsonlines
from run import training_step, map_losses
from utils.losses import FocalLoss, MarginFocalLoss, RocStar
from torch import nn, optim
from transformers import get_linear_schedule_with_warmup
import torch
import pickle 
import bottom_up


def write_submission(model, test_dataloader):
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    confidences = []
    labels = []
    ids = []
    with torch.set_grad_enabled(False):
        
        for i, batch in enumerate(tqdm(test_dataloader, unit="batch")): 
            logits = model(input_ids=batch[0].to(device), attention_mask=batch[1].to(device), visual_embeddings=batch[3].to(device))
            # probs = F.softmax(logits, dim=1)
            probs = torch.sigmoid(logits)
            # confidences += probs[:, 1].cpu().tolist()
            confidences += logits.cpu().tolist()
            # labels += probs.max(dim=1)[1].cpu().tolist()
            labels += (probs >= .4).cpu().tolist()
            ids += batch[4].tolist()
    
    return ids, confidences, labels

  
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--features', default='mask_rcnn_features')
    parser.add_argument('--loss', default='margin')
    
    args = parser.parse_args()
    models = [] 
    list_confidences = []

    test_set = []
    with jsonlines.open('data/test.jsonl', 'r') as f:
      for line in f: 
        test_set.append(line)

    dev_set = []
    with jsonlines.open('data/dev.jsonl', 'r') as f:
      for line in f: 
        dev_set.append(line)

    print('building dataloaders ...')
    with open(f'{args.features}/images_features_dict.pkl', 'rb') as f:
      images_features_dict = pickle.load(f)
    test_dataloader = create(data=test_set, datatype='test', batch_size=32, images_features_dict=images_features_dict)
    dev_dataloader = create(data=dev_set, datatype='dev', batch_size=16, images_features_dict=images_features_dict)
    print('done !')

    for filename in os.listdir('saved_models'):
        if 'ensemble' in filename:
          model = MyVisualBert()
          model.load_state_dict(torch.load(f'saved_models/{filename}'))
          # finetune on the dev set for one epoch 
          optimizer = optim.AdamW(model.parameters(), lr=1e-5, eps=1e-8)
          total_steps = 2 * len(dev_dataloader) 
          scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
          criterion = map_losses[args.loss]()
          device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
          model = model.to(device)
          
          training_step(0, optimizer, dev_dataloader, model, criterion, scheduler, device, args.loss)

          ids, confidences, _ = write_submission(model, test_dataloader)
          list_confidences.append(confidences)
    
    confidences = np.mean(list_confidences, axis=0)
    confidences = torch.sigmoid(torch.tensor(confidences)).numpy()
    # list_confidences = np.transpose(list_confidences)
    # confidences = np.zeros(list_confidences.shape[0])
    # for i, elt in enumerate(list_confidences):
      
    #     if all(c < .3 for c in elt):
    #         confidences[i] = np.min(elt)
    #     elif all(c > .7 for c in elt):
    #         confidences[i] = np.max(elt)
    #     else:
    #         confidences[i] = np.mean(elt)

    labels = (confidences >= .5).astype(int)

    df = pd.DataFrame({'id': ids, 'proba': confidences, 'label': labels})
    df.to_csv('new_features.csv', index=False)






