import pickle
import random
import numpy as np
import torch
import torch.nn as nn
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


def get_train_test_split_seq2seq():
    chinese = pickle.load(open("../data/chinese_lines.pk", "rb"))
    english = pickle.load(open("../data/english_lines.pk", "rb"))
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
            yield padded_english, torch.tensor(length_english, dtype=torch.int64), padded_chinese, torch.tensor(length_chinese, dtype=torch.int64)
