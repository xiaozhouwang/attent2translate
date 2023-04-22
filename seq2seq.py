import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pickle


HIDDENSIZE = 256
DEVICE = torch.device("cuda")

def get_vocab_size():
    token_maps = pickle.load(open("../data/token_map.pk", "rb"))
    return {'chinese': len(token_maps[0]), 'english': len(token_maps[1])}
vocabs = get_vocab_size()

class Encoder(nn.Module):
    """encode the source sentence: english"""
    def __init__(self, input_embed_size=128, hidden_size=HIDDENSIZE, vocab_size=vocabs['english'], bidirectional=True):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size+1, input_embed_size) # +1 for since we also need padding index
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_embed_size, hidden_size, num_layers=1, bidirectional=bidirectional, batch_first=True)

    def forward(self, input_x, lengths):
        x = self.embedding(input_x)
        x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(x_packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        return output, hidden, cell


class NoAttention(nn.Module):
    """no attention, just take the last output of encoder as the attention output"""
    def __init__(self):
        super(NoAttention, self).__init__()

    def forward(self, output, encoder_outputs, encoder_sequence_lengths):
        return torch.stack([x[-1] for x in nn.utils.rnn.unpad_sequence(encoder_outputs, encoder_sequence_lengths,
                                                                       batch_first=True)])


class AdditiveAttention(nn.Module):
    """additive attention"""
    def __init__(self, hidden_size=HIDDENSIZE*2):
        super(AdditiveAttention, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, output, encoder_outputs, encoder_sequence_lengths):
        query = output
        key = encoder_outputs
        value = encoder_outputs
        mask = create_mask_from_lengths(encoder_sequence_lengths)
        key_proj = self.W1(key)
        query_proj = self.W2(query)
        energy = self.V(torch.tanh(key_proj + query_proj))
        energy = energy.masked_fill(mask == 0, -1e10)
        attention_weights = torch.softmax(energy, dim=1)
        context_vector = torch.sum(attention_weights * value, dim=1)
        return context_vector


def create_mask_from_lengths(encoder_seq_lengths, max_length=None, device=DEVICE):
    if max_length is None:
        max_length = encoder_seq_lengths.max().item()
    mask = torch.arange(max_length).expand(len(encoder_seq_lengths), max_length) < encoder_seq_lengths.unsqueeze(1)
    return mask.unsqueeze(-1).to(device)


class OneStepDecoder(nn.Module):
    """decoding one step a time"""
    def __init__(self, output_embed_size, vocab_size, hidden_size=HIDDENSIZE*2):
        super(OneStepDecoder, self).__init__()
        self.output_size = vocab_size+1 # add padding
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(output_embed_size, hidden_size, num_layers=1, batch_first=True)
        self.attention = AdditiveAttention()
        self.out = nn.Linear(hidden_size, self.output_size)

    def forward(self, x, hidden, cell, encoder_outputs, encoder_sequence_lengths):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        context_vector = self.attention(output, encoder_outputs, encoder_sequence_lengths)
        assert context_vector.size() == (x.size(0), self.hidden_size)
        output = output + context_vector.unsqueeze(1)
        output = self.out(output)
        return output, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size=vocabs['chinese'], output_embed_size=128, device=DEVICE):
        super(Seq2Seq, self).__init__()
        self.target_embedding = nn.Embedding(vocab_size + 1, output_embed_size)
        self.output_embed_size = output_embed_size
        self.encoder = Encoder()
        self.decoder = OneStepDecoder(output_embed_size, vocab_size)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=0)
        self.device = device

    def forward(self, input_sequence, input_seq_lengths, output_sequence):
        encoder_outputs, hidden, cell = self.encoder(input_sequence, input_seq_lengths)
        outputs = []
        max_output_sequence = output_sequence.size(1)
        decoder_input = torch.zeros((input_sequence.size(0), 1, self.output_embed_size), device=self.device)
        hidden = torch.zeros((1, input_sequence.size(0), HIDDENSIZE*2), device=self.device)
        cell = torch.zeros((1, input_sequence.size(0), HIDDENSIZE*2), device=self.device)
        output_embed = self.target_embedding(output_sequence)
        for t in range(max_output_sequence):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs, input_seq_lengths)
            decoder_input = output_embed[:, t].unsqueeze(1)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=1)
        loss = self.loss_func(outputs.transpose(-2, -1), output_sequence)
        return outputs, loss

    def translate(self, input_sequence, input_seq_lengths, max_output_sequence=100):
        encoder_outputs, _, _ = self.encoder(input_sequence, input_seq_lengths)
        outputs = []
        decoder_input = torch.zeros((input_sequence.size(0), 1, self.output_embed_size), device=self.device)
        hidden = torch.zeros((1, input_sequence.size(0), HIDDENSIZE*2), device=self.device)
        cell = torch.zeros((1, input_sequence.size(0), HIDDENSIZE*2), device=self.device)
        for t in range(max_output_sequence):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs, input_seq_lengths)
            decoder_input = self.target_embedding(output.max(dim=2)[1]) # top 1 prob
            outputs.append(output)
        outputs = torch.cat(outputs, dim=1)
        return outputs