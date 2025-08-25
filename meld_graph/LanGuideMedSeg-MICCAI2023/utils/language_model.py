import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch.nn as nn
from transformers import AutoModel


class BERTModel(nn.Module):
    def __init__(self, bert_type, project_dim, tokenizer):
        super(BERTModel, self).__init__()

        self.model = AutoModel.from_pretrained(
            bert_type, output_hidden_states=True
        )  # ,trust_remote_code=True)

        # if tokenizer is not None:
        #     new_vocab_size = len(tokenizer)
        #     # resize both token embeddings и позиционные эмбеддинги
        #     self.model.resize_token_embeddings(new_vocab_size)

        # 1) freeze the parameters
        for p in self.model.parameters():
            p.requires_grad = False

        # 2) unfreeze_last_k слоёв BERT
        #    BertEncoder store them in .encoder.layer: list of 12 BertLayer
        for layer in self.model.encoder.layer[-3:]:
            for p in layer.parameters():
                p.requires_grad = True

        # 3) And pooler (for fine-tuning outputs)
        for p in self.model.pooler.parameters():
            p.requires_grad = True

    def forward(self, input_ids, attention_mask):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = output.last_hidden_state
        return {"feature": last_hidden}  # embed
        # get 1+2+last layer
        # last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2], output['hidden_states'][-1]]) # n_layer, batch, seqlen, emb_dim
        # embed = last_hidden_states.permute(1,0,2,3).mean(2).mean(1) # pooling
        # embed = self.project_head(embed)

        # return {'feature':output['hidden_states'],'project': None} # embed
