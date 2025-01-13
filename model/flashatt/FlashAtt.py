'''
substitute TransformerEncoder to Blocks in flash_attn_func.py
'''

import torch.nn.functional as F
import torch.nn as nn
from argparse import Namespace
from flash_attn.models.bert import BertConfig, BertEncoder
#from model.flashatt.flashatt_layer import BertEncoder, BertConfig

class FalshAttEncoder(nn.Module):
    def __init__(self, param: Namespace = Namespace(), simple = False):
        super(FalshAttEncoder, self).__init__()
        self.p = param
        encoder_config = BertConfig(
            hidden_size=self.p.embed_dim,
            num_hidden_layers=self.p.trans_layer,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
            use_flash_attn=True,
            num_attention_heads = self.p.num_heads,
            intermediate_size = self.p.hidden_dim,
            pad_token_id=0
        )
        self.encoder = BertEncoder(encoder_config)

    def forward(self, batch_embedding, batch_pad_mask = None):
        '''
        batch_embedding: [batch, len, dim]
        batch_pad_mask: [batch, len], True for no padding
        
        Return: 
        x_seq: [batch, len, dim]
        attn_score: a list of n_layers * [batch, num_heads, len, len]
        '''
        x_seq = self.encoder(batch_embedding, key_padding_mask=batch_pad_mask) #[batch, len, dim]
        return x_seq
