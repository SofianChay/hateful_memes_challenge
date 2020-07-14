import pandas as pd 
import torch
import torch.nn.functional as F
import os 
import argparse 
from model import MyVisualBert
from tqdm import tqdm
from dataset import create 
import numpy as np
import jsonlines


def write_submission(model, test_dataloader):
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    confidences = []
    labels = []
    ids = []
    with torch.set_grad_enabled(False):
        
        for i, batch in enumerate(tqdm(test_dataloader, unit="batch")): 
            logits = model(input_ids=batch[0].to(device), attention_mask=batch[1].to(device), visual_embeddings=batch[3].to(device))
            probs = F.softmax(logits, dim=1)
            confidences += probs[:, 1].cpu().tolist()
            labels += probs.max(dim=1)[1].cpu().tolist()
            ids += batch[4].tolist()
    
    return ids, confidences, labels

  
if __name__ == '__main__':

    models = [] 
    list_confidences = []

    test_set = []
    with jsonlines.open('data/test.jsonl', 'r') as f:
      for line in f: 
        test_set.append(line)

    test_dataloader = create(data=test_set, datatype='test', batch_size=32)
    
    for filename in os.listdir('saved_models'):
        if 'ensemble' in filename and 'ter' in filename:
          model = MyVisualBert()
          model.load_state_dict(torch.load(f'saved_models/{filename}'))
          ids, confidences, _ = write_submission(model, test_dataloader)
          list_confidences.append(confidences)
    
    confidences = np.mean(list_confidences, axis=0)
    labels = (confidences >= .4).astype(int)

    df = pd.DataFrame({'id': ids, 'proba': confidences, 'label': labels})
    df.to_csv('new_features.csv', index=False)






