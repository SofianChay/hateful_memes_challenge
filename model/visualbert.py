import torch.nn as nn
import torch.nn.functional as F
import torch
from mmf.common.registry import registry


model_cls = registry.get_model_class("visual_bert")


class MyVisualBert(nn.Module):

    def __init__(self):
        super(MyVisualBert, self).__init__()
        self.visual_bert = model_cls.from_pretrained(
            "visual_bert.finetuned.hateful_memes.from_coco").model
        self.visual_bert.output_hidden_states = True
        # self.visual_bert = model_cls.from_pretrained("visual_bert.pretrained.coco").model
        self.image_features = nn.Sequential(
            nn.Linear(2048, 2048), nn.ReLU(), nn.LayerNorm(2048))
        self.classifier = nn.Sequential(nn.Linear(768, 768), nn.ReLU(
        ), nn.Dropout(.1), nn.LayerNorm(768), nn.Linear(768, 1))

    def forward(self, input_ids, attention_mask, visual_embeddings):

        device = input_ids.device
        visual_embeddings = self.image_features(visual_embeddings)
        _, pooled, _ = self.visual_bert.bert(input_ids=input_ids,  # tokens
                                             attention_mask=attention_mask,  # attention mask phrase + mot
                                             visual_embeddings=visual_embeddings,  # 2048
                                             visual_embeddings_type=torch.ones((input_ids.shape[0], visual_embeddings.shape[1]), dtype=torch.long).to(device))

        logits = self.classifier(pooled)
        return logits.squeeze(-1)


class CustomPooler(nn.Module):
    """
    Custom RoBERTa Pooling head that takes the n last layers.
    """

    def __init__(self, input_size=768, output_size=768):
        """
        Constructor.

        Parameters:
            nb_layers {int}: number of layers to consider (default: {1}).
            input_size {int}: size of the input features (default: {768}).
            output_size {int}: size of the output features (default: {1536}).
        """
        super(CustomPooler, self).__init__()

        self.input_size = input_size
        self.pooler = nn.LSTM(input_size,
                              hidden_size=output_size//2,
                              batch_first=True,
                              bidirectional=True,
                              num_layers=2)

    def forward(self, hidden_states):
        """
        Usual torch forward function.

        Parameters:
            hidden_states {list of torch tensors}: hidden states of the model, the last one being at index 0.

        Returns:
            torch tensor: pooled features.
            torch tensor: hidden states of the considered layers.
        """
        pooled = torch.tanh(self.pooler(hidden_states)[0])
        return pooled


class VisualBertWithPooler(nn.Module):
    def __init__(self, dropout=.1, pooler_ft=768):
        super(VisualBertWithPooler, self).__init__()
        self.visual_bert = model_cls.from_pretrained(
            "visual_bert.pretrained.coco").model
        # self.visual_bert = model_cls.from_pretrained("visual_bert.pretrained.coco").model
        # self.pooler = CustomPooler(input_size=768, output_size=pooler_ft)
        self.pooler = nn.Sequential(nn.Linear(768, 768), nn.Tanh(),)

        self.intermediate_1 = nn.Linear(pooler_ft,
                                        pooler_ft)
        self.intermediate_2 = nn.Linear(pooler_ft, pooler_ft//2)
        self.drop_1 = nn.Dropout(dropout)
        self.drop_2 = nn.Dropout(dropout)
        self.logits = nn.Linear(pooler_ft//2, 1)

    def forward(self, input_ids, attention_mask, visual_embeddings):

        device = input_ids.device
        hs, _, _ = self.visual_bert.bert(input_ids=input_ids,  # tokens
                                         attention_mask=attention_mask,  # attention mask phrase + mot
                                         visual_embeddings=visual_embeddings,  # 2048
                                         visual_embeddings_type=torch.ones((input_ids.shape[0], visual_embeddings.shape[1]), dtype=torch.long).to(device))
        hs = hs.mean(1)
        x = self.pooler(hs)
        x = self.drop_1(x)
        x = torch.tanh(self.intermediate_1(x) + hs)
        x = torch.tanh(self.intermediate_2(x))
        x = self.drop_2(x)
        logits = self.logits(x).squeeze(-1)
        return logits
