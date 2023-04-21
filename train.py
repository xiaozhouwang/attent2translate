import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
import pickle


def get_vocab_size():
    token_maps = pickle.load(open("../data/token_map.pk", "rb"))
    return {'chinese': len(token_maps[0]), 'english': len(token_maps[1])}
vocabs = get_vocab_size()

class DataLoader:
    def __init__(self, batch_size, shuffle=True):
        self.chinese = pickle.load(open("../data/chinese_lines.pk", "rb"))
        self.english = pickle.load(open("../data/english_lines.pk", "rb"))
        self.batch_size = batch_size
        self.num_samples = len(self.chinese)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            index_list = list(range(len(self.chinese)))
            self.chinese = [self.chinese[i] for i in index_list]
            self.english = [self.english[i] for i in index_list]
        for i in range(0, self.num_samples, self.batch_size):
            """"""
            batch_chinese = self.chinese[i:i+self.batch_size]
            batch_english = self.english[i:i+self.batch_size]
            length_chinese = [len(x) for x in batch_chinese]
            length_english = [len(x) for x in batch_english]
            padded_chinese = nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in batch_chinese], batch_first=True,
                                                       padding_value=0)
            padded_english = nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in batch_english], batch_first=True,
                                                       padding_value=0)
            yield padded_english, length_english, torch.tensor(padded_chinese), torch.tensor(length_chinese)


class Encoder(nn.Module):
    """encode the source sentence: english"""
    def __init__(self, input_embed_size=256, hidden_size=512, vocab_size=vocabs['english'], bidirectional=True):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size+1, input_embed_size) # +1 for since we also need padding index
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_embed_size, hidden_size, num_layers=2, bidirectional=bidirectional, batch_first=True)

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

    def forward(self, encoder_outputs, encoder_sequence_lengths):
        return torch.stack([x[-1] for x in nn.utils.rnn.unpad_sequence(encoder_outputs, encoder_sequence_lengths,
                                                                       batch_first=True)])


class OneStepDecoder(nn.Module):
    """decoding one step a time"""
    def __init__(self, output_embed_size=256, hidden_size=1024, vocab_size=vocabs['chinese']):
        super(OneStepDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, output_embed_size)
        self.output_size = vocab_size+1 # add padding
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(output_embed_size, hidden_size, num_layers=1, batch_first=True)
        self.attention = NoAttention()
        self.out = nn.Linear(hidden_size, self.output_size)

    def forward(self, input_x, hidden, cell, encoder_outputs, encoder_sequence_lengths):
        x = self.embedding(input_x)
        if hidden is None:
            output, (hidden, cell) = self.lstm(x)
        else:
            output, (hidden, cell) = self.lstm(x, (hidden, cell))
        context_vector = self.attention(encoder_outputs, encoder_sequence_lengths)
        assert context_vector.size() == (input_x.size(0), self.hidden_size)
        output = output + context_vector.unsqueeze(1)
        output = self.out(output)
        return output, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder()
        self.decoder = OneStepDecoder()
        self.loss_func = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, input_sequence, input_seq_lengths, output_sequence=None, max_output_sequence=None):
        encoder_outputs, _, _ = self.encoder(input_sequence, input_seq_lengths)
        outputs = []
        max_output_sequence = output_sequence.size(1) if output_sequence is not None else max_output_sequence
        for t in range(max_output_sequence):
            decoder_input = output_sequence[:, t].unsqueeze(1)
            if t == 0:
                output, hidden, cell = decoder(decoder_input, None, None, encoder_outputs, input_seq_lengths)
            else:
                output, hidden, cell = decoder(decoder_input, hidden, cell, encoder_outputs, input_seq_lengths)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=1)
        loss = self.loss_func(outputs.transpose(-2, -1), output_sequence) if output_sequence is not None else None
        return outputs, loss


if __name__ == '__main__':
    data_loader = DataLoader(batch_size=100)
    encoder = Encoder(64, 128, vocabs['english'])
    decoder = OneStepDecoder(64, 128*2, vocabs['chinese'])
    # Iterate over the batches
    for batch_idx, batch in enumerate(data_loader):
        padded_english, length_english, padded_chinese, length_chinese = batch
        encoder_output, _, _ = encoder(padded_english, length_english)
        for t in range(padded_chinese.size(1)): # decode one step a time!
            decoder_input = padded_chinese[:, t].unsqueeze(1)
            if t == 0:
                output, hidden, cell = decoder(decoder_input, None, None, encoder_output, length_english)
                print(output.shape, hidden.shape, cell.shape)
            else:
                output, hidden, cell = decoder(decoder_input, hidden, cell, encoder_output, length_english)
                print(output.shape, hidden.shape, cell.shape)
        break
