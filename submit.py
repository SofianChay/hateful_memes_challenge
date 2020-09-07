import jsonlines
import pickle
from dataset import dataset
import os
from model.visualbert import MyVisualBert, VisualBertWithPooler
import torch
from model import engine
import numpy as np
import pandas as pd
# Create test dataset
if __name__ == '__main__':

    models = []
    list_confidences = []

    test_set = []
    with jsonlines.open('data/test.jsonl', 'r') as f:
        for line in f:
            test_set.append(line)
    print('building dataloaders ...')
    with open(f'mask_rcnn_features/images_features_dict.pkl', 'rb') as f:
        images_features_dict = pickle.load(f)
    test_dataloader = dataset.create(data=test_set, datatype='test',
                                     batch_size=32, images_features_dict=images_features_dict)
    print('done !')
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
    print(f'Working on {device}')
    all_preds = []
    for filename in os.listdir('saved_models'):
        model = VisualBertWithPooler()
        model.to(device)
        model.load_state_dict(torch.load(f'saved_models/{filename}'))
        ids, predictions = engine.predict(test_dataloader, model, device)
        all_preds.append(predictions)
    proba = np.concatenate(all_preds, 1).mean(1)
    labels = (proba >= .4).astype(int)

    df = pd.DataFrame({'id': ids, 'proba': proba, 'label': labels})
    df.to_csv('submission.csv', index=False)
