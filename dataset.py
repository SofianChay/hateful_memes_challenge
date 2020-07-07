import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import Dataset
import pickle
from transformers import BertTokenizer
from transformers import BertConfig


config = BertConfig.from_pretrained('bert-base-uncased')
tkz = BertTokenizer.from_pretrained('bert-base-uncased')

class MyDataset(Dataset):

    def __init__(self, input_ids, attention_masks, images_features, labels, ids):
        self.input_ids = torch.tensor(input_ids)
        self.attention_masks = torch.tensor(attention_masks)
        self.images_features = images_features
        self.labels = labels
        self.ids = ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.attention_masks[index], self.labels[index], self.images_features[index], self.ids[index]



def create(data, datatype, batch_size):

    input_ids = []
    attention_masks = []
    ids = []
    labels = []
    max_len = 0

    for elt in data:
        inputs = tkz.encode(elt['text'])
        max_len = max(max_len, len(inputs))
        input_ids.append(inputs)
        attention_masks.append([1] * len(inputs))
        ids.append(elt['id'])
        if datatype == 'test':
            labels.append(-1)
        else:
            labels.append(elt['label'])

    with open(f'image_features/images_features_dict.pkl', 'rb') as f:
        images_features_dict = pickle.load(f)

    images_features_list = []
    for elt in data:
      images_features_list.append(images_features_dict[elt['id']])

    max_len_features = max([len(features) for features in images_features_list])
    attention_masks_images = []

    for i in range(len(images_features_list)):
        attention_masks_images.append([1] * len(images_features_list[i]) + [0] * (max_len_features  - len(images_features_list[i])))
        images_features_list[i] = torch.cat((images_features_list[i], torch.zeros((max_len_features - len(images_features_list[i]), 1024))), dim=0)

    for i in range(len(data)):
        input_ids[i] += [config.pad_token_id] * (max_len - len(input_ids[i]))
        attention_masks[i] += [0] * (max_len - len(attention_masks[i])) + attention_masks_images[i]

    dataset = MyDataset(input_ids, attention_masks, images_features_list, labels, ids)

    if datatype == 'train':
        dataloader =  DataLoader(
                                dataset,  
                                sampler=RandomSampler(dataset), 
                                batch_size=batch_size 
                            )
    else:
        dataloader =  DataLoader(
                        dataset,  
                        sampler=SequentialSampler(dataset), 
                        batch_size=batch_size 
                    )

    return dataloader
