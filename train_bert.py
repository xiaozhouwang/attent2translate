"""Simple Bert classification model, with only [CLS] token as first token"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle
from tqdm import tqdm
import wandb
from transformer import EncoderTransformerLayer, create_src_mask


class Dataloader:
    def __init__(self, seq_src, seq_pos, seq_neg, batch_size, shuffle=True):
        assert len(seq_pos) == len(seq_neg) == len(seq_src)
        self.num_samples = len(seq_pos)
        self.seq_pos = seq_pos
        self.seq_neg = seq_neg
        self.seq_src = seq_src
        self.seq_src = [[1]+s for s in self.seq_src] # add [CLS] token
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


class BERT(nn.Module):
    def __init__(self, input_dim=44800, hidden_size=256, num_layers=3, num_heads=8, ff_size=512, dropout_rate=0, max_length=201):
        super(BERT, self).__init__()
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


class Model(nn.Module):
    def __init__(self, hidden_size=256):
        super(Model, self).__init__()
        self.bert = BERT(hidden_size=hidden_size)
        self.fc = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, 1))

    def forward(self, src, seg_seq, src_mask):
        x = self.bert(src, seg_seq, src_mask)
        x = self.fc(x.mean(dim=1))
        return x.flatten()


if __name__ == '__main__':
    run_name = "bert classification"
    wandb.init(project="Classification_Bert_GPT", name=run_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NEPOCH = 100
    BestEpoch=0
    BestLoss = np.Inf
    data = pickle.load(open("../data/data_classification.pk", "rb"))
    train_data_loader = Dataloader(seq_src=data['train']['src'], seq_pos=data['train']['pos'],
                                   seq_neg=data['train']['neg'], batch_size=256)
    test_data_loader = Dataloader(seq_src=data['test']['src'], seq_pos=data['test']['pos'],
                                  seq_neg=data['test']['neg'], batch_size=256, shuffle=False)
    model = Model().to(device)
    optimizer = optim.AdamW(model.parameters(), fused=True)
    optimizer.zero_grad()
    loss_func = nn.BCEWithLogitsLoss()

    for epoch in range(NEPOCH):
        epoch_loss = 0
        for s, batch in enumerate(tqdm(train_data_loader)):
            seq, seq_seg, batch_tar = batch
            seq = seq.to(device)
            batch_tar = batch_tar.to(device)
            seq_seg = seq_seg.to(device)
            src_mask = create_src_mask(seq).to(device)
            outputs = model(seq, seq_seg, src_mask)
            loss = loss_func(outputs, batch_tar)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

            if s % 10 == 0:
                wandb.log({'epoch loss': loss.item()})

        epoch_loss /= len(train_data_loader)
        print(f"epoch: {epoch}, training loss: {epoch_loss}")
        wandb.log({'train_loss': epoch_loss})

        """test loss"""
        epoch_loss = 0
        with torch.no_grad():
            for batch in tqdm(test_data_loader):
                seq, seq_seg, batch_tar = batch
                seq = seq.to(device)
                batch_tar = batch_tar.to(device)
                seq_seg = seq_seg.to(device)
                src_mask = create_src_mask(seq).to(device)
                outputs = model(seq, seq_seg, src_mask)
                loss = loss_func(outputs, batch_tar)
                epoch_loss += loss.item()
            epoch_loss /= len(test_data_loader)
            print(f"epoch: {epoch}, test loss: {epoch_loss}")
            wandb.log({'test_loss': epoch_loss, 'epoch': epoch})

        if epoch_loss < BestLoss:
            BestLoss = epoch_loss
            BestEpoch = epoch + 1

        if epoch - 3 > BestEpoch:
            print(f"early stop at {epoch+1} with best epoch {BestEpoch} and test loss {BestLoss}.")
            break
