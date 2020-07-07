import torch.nn as nn
import torch.nn.functional as F
import torch
from mmf.common.registry import registry


model_cls = registry.get_model_class("visual_bert")


class MyVisualBert(nn.Module):

    def __init__(self):
        super(MyVisualBert, self).__init__()
        self.visual_bert = model_cls.from_pretrained("visual_bert.finetuned.hateful_memes.from_coco")
        self.visual_bert = self.visual_bert.model
        self.image_features = nn.Sequential(nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, 2048), nn.LayerNorm(2048))
        self.visual_bert.classifier = nn.Sequential(nn.Linear(768, 768), nn.Dropout(.1), nn.ReLU(), nn.LayerNorm(768), nn.Linear(768, 2))

    
    def forward(self, input_ids, attention_mask, visual_embeddings):

        device = input_ids.device
        visual_embeddings = self.image_features(visual_embeddings)
        return self.visual_bert(input_ids=input_ids, 
                                attention_mask=attention_mask, 
                                visual_embeddings=visual_embeddings, 
                                visual_embeddings_type=torch.ones((input_ids.shape[0], visual_embeddings.shape[1]), dtype=torch.long).to(device))['scores']
