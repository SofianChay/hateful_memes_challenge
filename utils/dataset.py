import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import Dataset
import pickle
from transformers import BertTokenizer, RobertaTokenizer
from transformers import BertConfig, RobertaConfig

# from fairseq.data.encoders.fastbpe import fastBPE
# from fairseq.data import Dictionary

import os
import numpy as np
from tqdm import tqdm 

# Load model
# config = RobertaConfig.from_pretrained(
#     "/content/drive/My Drive/hateful_memes/bertweet/BERTweet_base_transformers/config.json"
# )

# # Load BPE encoder
# # jsuis un pirate 
# class Args:
#   def __init__(self):
#     self.bpe_codes = "/content/drive/My Drive/hateful_memes/bertweet/BERTweet_base_transformers/bpe.codes"



# bpe = fastBPE(Args())

# Load the dictionary  
# vocab = Dictionary()
# vocab.add_from_file("/content/drive/My Drive/hateful_memes/bertweet/BERTweet_base_transformers/dict.txt")


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



def create(data, datatype, batch_size, images_features_dict):

    input_ids = []
    attention_masks = []
    ids = []
    labels = []
    max_len = 0

    for elt in data:
        inputs = tkz.encode(elt['text'])
        # subwords = '<s> ' + bpe.encode(elt['text']) + ' </s>'
        # inputs = vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()
        max_len = max(max_len, len(inputs))
        input_ids.append(inputs)
        attention_masks.append([1] * len(inputs))
        ids.append(elt['id'])
        if datatype == 'test':
            labels.append(-1)
        else:
            labels.append(elt['label'])
    print(f'maximum meme text length : {max_len}')

    images_features_list = []
    for elt in data:
      images_features_list.append(images_features_dict[elt['id']])
      
    max_len_features = max([len(features) for features in images_features_list])
    attention_masks_images = []

    for i in range(len(images_features_list)):
        attention_masks_images.append([1] * len(images_features_list[i]) + [0] * (max_len_features  - len(images_features_list[i])))
        images_features_list[i] = torch.cat((images_features_list[i], torch.zeros((max_len_features - len(images_features_list[i]), 2048))), dim=0)

    for i in range(len(images_features_list)):
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
