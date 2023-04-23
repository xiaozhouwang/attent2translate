import torch
import torch.nn as nn
import pickle


def get_vocab_size():
    token_maps = pickle.load(open("../data/token_map.pk", "rb"))
    return {'chinese': len(token_maps[0]), 'english': len(token_maps[1])}
vocabs = get_vocab_size()


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.Wq = nn.Linear(hidden_size, hidden_size)
        self.Wk = nn.Linear(hidden_size, hidden_size)
        self.Wv = nn.Linear(hidden_size, hidden_size)

        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.Wq(query)
        K = self.Wk(key)
        V = self.Wv(value)

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.head_dim ** 0.5
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))

        attention = torch.softmax(energy, dim=-1)
        out = torch.matmul(attention, V).permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, -1, self.hidden_size)
        out = self.fc(out)

        return out, attention


class PointwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, ff_size):
        super(PointwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_size, ff_size)
        self.fc2 = nn.Linear(ff_size, hidden_size)

    def forward(self, x):
        out = self.fc1(x)
        out = torch.relu(out)
        out = self.fc2(out)
        return out


class EncoderTransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_size, dropout_rate):
        super(EncoderTransformerLayer, self).__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ff = PointwiseFeedForward(hidden_size, ff_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        att_out, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(att_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class Encoder(nn.Module):
    def __init__(self, input_dim=vocabs['english']+1, hidden_size=256, num_layers=3, num_heads=8, ff_size=512, dropout_rate=0, max_length=100):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, hidden_size))
        self.layers = nn.ModuleList(
            [EncoderTransformerLayer(hidden_size, num_heads, ff_size, dropout_rate) for _ in range(num_layers)])

    def forward(self, src, src_mask=None):
        embedded = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        x = embedded
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class DecoderTransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_size, dropout_rate):
        super(DecoderTransformerLayer, self).__init__()
        self.self_attention = MultiHeadAttention(hidden_size, num_heads)
        self.cross_attention = MultiHeadAttention(hidden_size, num_heads)
        self.feed_forward = PointwiseFeedForward(hidden_size, ff_size)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, encoder_outputs, tgt_mask=None, src_mask=None):
        att_out, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(att_out))
        att_out, _ = self.cross_attention(x, encoder_outputs, encoder_outputs, src_mask)
        x = self.norm2(x + self.dropout(att_out))
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x


class Decoder(nn.Module):
    def __init__(self, output_dim=vocabs['chinese']+1, hidden_size=256, num_layers=3, num_heads=8, ff_size=512, dropout_rate=0, max_length=100):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, hidden_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, hidden_size))
        self.layers = nn.ModuleList(
            [DecoderTransformerLayer(hidden_size, num_heads, ff_size, dropout_rate) for _ in range(num_layers)])
        self.fc_out = nn.Linear(hidden_size, output_dim)

    def forward(self, tgt, encoder_outputs, tgt_mask=None, src_mask=None):
        embedded = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
        x = embedded
        for layer in self.layers:
            x = layer(x, encoder_outputs, tgt_mask, src_mask)
        logits = self.fc_out(x)
        return logits



class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_outputs = self.encoder(src, src_mask)
        logits = self.decoder(tgt, encoder_outputs, tgt_mask, src_mask)
        return logits


def create_src_mask(src, pad_idx=0):
    # src: (batch_size, src_len)
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    # src_mask: (batch_size, 1, 1, src_len)
    return src_mask


def create_tgt_mask(tgt, pad_idx=0, device='cuda'):
    # tgt: (batch_size, tgt_len)
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    # tgt_pad_mask: (batch_size, 1, 1, tgt_len)

    tgt_len = tgt.size(1)
    tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len))).bool().to(device)
    # tgt_sub_mask: (tgt_len, tgt_len)

    tgt_mask = tgt_pad_mask & tgt_sub_mask
    # tgt_mask: (batch_size, 1, tgt_len, tgt_len)
    return tgt_mask

