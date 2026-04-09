"""
Question encoder: pre-trained BERT for question intent.
Output: pooled question embedding and token embeddings for cross-attention.
"""
from typing import Tuple

import torch
import torch.nn as nn

try:
    from transformers import BertModel
except ImportError:
    BertModel = None


class QuestionEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        question_dim: int = 768,
    ):
        super().__init__()
        if BertModel is None:
            raise ImportError(
                "transformers is required for BERT-backed question encoder. Run: pip install transformers"
            )
        self.bert = BertModel.from_pretrained(model_name)
        
        # Freeze all BERT parameters first
        self.bert.requires_grad_(False)
        
        # Unfreeze the last 2 transformer layers + pooler for VQA adaptation
        # Keeps 10 layers frozen (stable representations) while 2 layers adapt to GQA
        num_layers = len(self.bert.encoder.layer)
        for layer in self.bert.encoder.layer[num_layers - 2:]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in self.bert.pooler.parameters():
            param.requires_grad = True
            
        self.hidden_size = self.bert.config.hidden_size  # 768 for base
        # Project to target dim if question_dim differs from BERT's hidden size
        if question_dim != self.hidden_size:
            self.proj = nn.Linear(self.hidden_size, question_dim)
        else:
            self.proj = None

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        input_ids: (B, L), attention_mask: (B, L)
        Returns: (B, hidden_size) pooled, (B, L, hidden_size) token embeddings
        """
        # Run with gradients (last 2 layers are trainable)
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output
        token_embeddings = out.last_hidden_state
        if self.proj is not None:
            pooled = self.proj(pooled)
            token_embeddings = self.proj(token_embeddings)
        return pooled, token_embeddings
