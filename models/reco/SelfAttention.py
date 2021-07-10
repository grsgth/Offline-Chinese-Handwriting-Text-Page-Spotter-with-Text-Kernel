import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, hidden):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.bias = nn.Parameter(torch.ones(hidden))
        self.variance_epsion = 1e-5

    def forward(self, x):
        # x 传入的shape是[L,N,E],对E做Norm
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsion)
        return self.weight * x + self.bias


class addAttention(nn.Module):
    # 构建shortcut和multi-head层，是attention block
    def __init__(self, input_dim, num_heads):
        super(addAttention, self).__init__()
        self.multihead = nn.MultiheadAttention(input_dim, num_heads)
        self.q_weight = nn.Linear(input_dim, input_dim)
        self.k_weight = nn.Linear(input_dim, input_dim)
        self.v_weight = nn.Linear(input_dim, input_dim)
        self.ln = LayerNorm(input_dim)

    def forward(self, input):
        output = input  # input shape: [L,N,E]
        q, k, v = self.q_weight(input), self.k_weight(input), self.v_weight(input)
        multi_attention, _ = self.multihead(q, k, v)  # (L, N, E)
        output += multi_attention
        output = self.ln(output)
        return output


class attentionLayer(nn.Module):
    def __init__(self, input_dim, hidden, num_heads, dropout=0.2):
        super(attentionLayer, self).__init__()
        self.liner1 = nn.Linear(input_dim, hidden)
        self.liner2 = nn.Linear(hidden, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention_block = addAttention(input_dim, num_heads)
        self.ln = LayerNorm(input_dim)

    def forward(self, input):
        attention_block = self.attention_block(input)
        output = attention_block
        feed_forward = self.liner2(self.dropout(F.relu(self.liner1(attention_block))))
        output += feed_forward
        return self.ln(output)


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden, num_heads,layernam, dropout=0.2):
        super(Decoder, self).__init__()
        layers=[attentionLayer(input_dim, hidden, num_heads,dropout)]*layernam
        self.decode = nn.Sequential(
            *layers
        )

    def forward(self, input):
        output = self.decode(input)
        return output
