import pickle
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from transformer import EncoderTransformerLayer
random.seed(124)


def preprocess_sentences(english, chinese, min_len=10, max_len=100):
    new_english = []
    new_chinese = []
    for e, c in zip(english, chinese):
        if len(e) < min_len or len(c) < min_len or len(e) > max_len or len(c) > max_len:
            continue
        new_chinese.append(c)
        new_english.append(e)
    return new_english, new_chinese


def get_train_test_split(chinese=None, english=None):
    chinese = pickle.load(open("../data/chinese_lines.pk", "rb")) if chinese is None else chinese
    english = pickle.load(open("../data/english_lines.pk", "rb")) if english is None else english
    test_index = set(random.sample(range(len(chinese)), int(0.1*len(chinese))))
    train_index = set(range(len(chinese))).difference(test_index)
    train_english = [english[i] for i in range(len(english)) if i in train_index]
    test_english = [english[i] for i in range(len(english)) if i in test_index]
    train_chinese = [chinese[i] for i in range(len(chinese)) if i in train_index]
    test_chinese = [chinese[i] for i in range(len(chinese)) if i in test_index]
    train_english, train_chinese = preprocess_sentences(train_english, train_chinese)
    test_english, test_chinese = preprocess_sentences(test_english, test_chinese)
    print("training size:", len(train_english), "test size:", len(test_english))
    return train_english, train_chinese, test_english, test_chinese


class S2SDataLoader:
    def __init__(self, seq_chinese, seq_english, batch_size, shuffle=True):
        self.chinese = seq_chinese
        self.english = seq_english
        self.batch_size = batch_size
        self.num_samples = len(self.chinese)
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(len(self.chinese)/self.batch_size))

    def __iter__(self):
        if self.shuffle:
            index_list = list(range(len(self.chinese)))
            random.shuffle(index_list)
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
            yield padded_english, torch.tensor(length_english, dtype=torch.int64), padded_chinese, torch.tensor(
                length_chinese, dtype=torch.int64)


def create_negative_samples(positive_samples):
    negative_samples = []
    for s in tqdm(positive_samples):
        l = len(s)
        longer_seqs = [lst for lst in positive_samples if len(lst) > l]
        shorter_seqs = [lst for lst in positive_samples if len(lst) < l]
        candidates = []
        if longer_seqs:
            candidates.append(random.choice(longer_seqs)[:l])
        if shorter_seqs:
            sampled_seq = random.choice(shorter_seqs)
            candidates.append(sampled_seq+s[len(sampled_seq):])
        negative_samples.append(random.choice(candidates))
    """check if negative samples is a list of lists with integers to ensure no funny bug"""
    assert all(isinstance(sublist, list) and all(isinstance(item, int) for item in sublist) for sublist in negative_samples)
    """check if lengths match"""
    assert [len(x) for x in negative_samples] == [len(x) for x in positive_samples]
    return negative_samples


class BertorGPTDataloader:
    def __init__(self, seq_src, seq_pos, seq_neg, batch_size, shuffle=True):
        assert len(seq_pos) == len(seq_neg) == len(seq_src)
        self.num_samples = len(seq_pos)
        self.seq_pos = seq_pos
        self.seq_neg = seq_neg
        self.seq_src = seq_src
        #self.seq_src = [[1]+s for s in self.seq_src] # add [CLS] token
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.num_samples/self.batch_size))

    def __iter__(self):
        if self.shuffle:
            index_list = list(range(len(self.seq_src)))
            random.shuffle(index_list)
            self.seq_pos = [self.seq_pos[i] for i in index_list]
            self.seq_neg = [self.seq_neg[i] for i in index_list]
            self.seq_src = [self.seq_src[i] for i in index_list]

        for i in range(0, self.num_samples, self.batch_size):
            batch_seq_pos = self.seq_pos[i:i+self.batch_size]
            batch_seq_neg = self.seq_neg[i:i+self.batch_size]
            batch_seq_src = self.seq_src[i:i+self.batch_size]

            padded_seq_pos = nn.utils.rnn.pad_sequence([torch.tensor(s) for s in batch_seq_pos], batch_first=True,
                                                       padding_value=0)
            padded_seq_neg = nn.utils.rnn.pad_sequence([torch.tensor(s) for s in batch_seq_neg], batch_first=True,
                                                       padding_value=0)
            padded_seq_src = nn.utils.rnn.pad_sequence([torch.tensor(s) for s in batch_seq_src], batch_first=True,
                                                       padding_value=0)
            seq_src_seg = torch.ones_like(padded_seq_src).masked_fill(padded_seq_src != 0, 1)

            """concat src, first half pos, and second half neg"""
            half_bs = padded_seq_neg.size(0)//2
            seq = torch.cat((padded_seq_src, torch.cat((padded_seq_pos[:half_bs], padded_seq_neg[half_bs:]), dim=0)),
                            dim=1)
            assert seq.shape == (padded_seq_neg.size(0), padded_seq_neg.size(1)+padded_seq_src.size(1))
            seq_seg = torch.cat((seq_src_seg, torch.zeros_like(padded_seq_neg)), dim=1)
            assert seq_seg.shape == seq.shape
            batch_tar = torch.tensor([1]*half_bs+[0]*(padded_seq_neg.size(0)-half_bs), dtype=torch.float32)

            yield seq, seq_seg, batch_tar


class BERTorGPTBase(nn.Module):
    def __init__(self, input_dim=44800, hidden_size=256, num_layers=3, num_heads=8, ff_size=512, dropout_rate=0, max_length=201):
        super(BERTorGPTBase, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, hidden_size))
        self.seg_embedding = nn.Embedding(2, hidden_size)
        self.layers = nn.ModuleList(
            [EncoderTransformerLayer(hidden_size, num_heads, ff_size, dropout_rate) for _ in range(num_layers)])

    def forward(self, src, seg_seq, src_mask=None):
        embedded = self.embedding(src) + self.positional_encoding[:, :src.size(1), :] + self.seg_embedding(seg_seq)
        x = embedded
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class BERTorGPT(nn.Module):
    def __init__(self, hidden_size=256):
        super(BERTorGPT, self).__init__()
        self.bert = BERTorGPTBase(hidden_size=hidden_size)
        self.fc = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, 1))

    def forward(self, src, seg_seq, src_mask):
        x = self.bert(src, seg_seq, src_mask)
        x = self.fc(x.mean(dim=1)) # without pretraining, it seems mean pooling is better than [CLS] token
        return x.flatten()
