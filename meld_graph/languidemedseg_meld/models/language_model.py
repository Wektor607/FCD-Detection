import os
import sys
from typing import Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
from transformers import AutoModel


class BERTModel(nn.Module):
    def __init__(self, 
        bert_type: str, 
        num_unfreeze_layers: int = 0, 
        use_pooler: bool = True
    ) -> None:
        super().__init__()

        self.model = AutoModel.from_pretrained(bert_type, output_hidden_states=True)

        # 1) freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # 2) unfreeze_last_k слоёв BERT
        #    BertEncoder store them in .encoder.layer: list of 12 BertLayer
        k = max( 0, min(num_unfreeze_layers, getattr(self.model.config, "num_hidden_layers", 12)))
        if (
            k > 0
            and hasattr(self.model, "encoder")
            and hasattr(self.model.encoder, "layer")
        ):
            for layer in self.model.encoder.layer[-k:]:
                layer.requires_grad_(True)
            
            # 3) And pooler (for fine-tuning outputs)
            self.use_pooler = (
                use_pooler
                and hasattr(self.model, "pooler")
                and self.model.pooler is not None
            )
            if self.use_pooler:
                self.model.pooler.requires_grad_(True)


    def forward(self, text: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = self.model(
            **text,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = output.last_hidden_state
        return {"feature": last_hidden}
