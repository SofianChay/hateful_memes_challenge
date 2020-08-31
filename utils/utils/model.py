import torch.nn as nn
import torch.nn.functional as F
import torch
from mmf.common.registry import registry


model_cls = registry.get_model_class("visual_bert")


class MyVisualBert(nn.Module):

    def __init__(self):
        super(MyVisualBert, self).__init__()
        self.visual_bert = model_cls.from_pretrained("visual_bert.pretrained.coco").model
        # self.image_features = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), nn.LayerNorm(2048))
        self.classifier_hateful = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Dropout(.1), nn.LayerNorm(768), nn.Linear(768, 1))
    
    def forward(self, input_ids, attention_mask, visual_embeddings):

        device = input_ids.device
        # visual_embeddings = self.image_features(visual_embeddings)
        embs, pooled, _ = self.visual_bert.bert(input_ids=input_ids, # tokens
                                attention_mask=attention_mask, # attention mask phrase + mot 
                                visual_embeddings=visual_embeddings, # 2048
                                visual_embeddings_type=torch.ones((input_ids.shape[0], visual_embeddings.shape[1]), dtype=torch.long).to(device))
        
        # embs = torch.sum(embs * attention_mask.unsqueeze(-1), dim=1) /torch.sum(attention_mask.unsqueeze(-1), dim=1)
        logits = self.classifier_hateful(pooled)

        return logits.squeeze(-1)
