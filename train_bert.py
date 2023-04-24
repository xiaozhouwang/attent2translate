"""Simple Bert classification model, with only [CLS] token as first token"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle
from tqdm import tqdm
import wandb
from transformer import Encoder, create_src_mask


class Dataloader:
    def __init__(self, seq_pos, seq_neg, batch_size, shuffle=True):
        self.num_samples = len(seq_pos) + len(seq_neg)
        assert len(seq_pos) == len(seq_neg)
        self.seq = seq_pos + seq_neg
        self.seq = [[1]+s for s in self.seq] # add [CLS] token
        self.batch_size = batch_size
        self.targets = [1] * len(seq_pos) + [0] * len(seq_neg)
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.num_samples/self.batch_size))

    def __iter__(self):
        if self.shuffle:
            index_list = list(range(len(self.seq)))
            random.shuffle(index_list)
            self.seq = [self.seq[i] for i in index_list]
            self.targets = [self.targets[i] for i in index_list]
        for i in range(0, self.num_samples, self.batch_size):
            batch_seq = self.seq[i:i+self.batch_size]
            batch_tar = self.targets[i:i+self.batch_size]
            padded_seq = nn.utils.rnn.pad_sequence([torch.tensor(s) for s in batch_seq], batch_first=True,
                                                       padding_value=0)
            yield padded_seq, torch.tensor(batch_tar, dtype=torch.float32)


class BERT(nn.Module):
    def __init__(self, hidden_size=512):
        super(BERT, self).__init__()
        self.encoder = Encoder(input_dim=44800, hidden_size=hidden_size, num_layers=6, num_heads=8, ff_size=hidden_size*2,
                               dropout_rate=0, max_length=201)
        self.outlayer = nn.Linear(hidden_size, 1)

    def forward(self, x, x_mask):
        x = self.encoder(src=x, src_mask=x_mask)
        return self.outlayer(x[:, 0, :]).flatten()


if __name__ == '__main__':
    run_name = "bert classification"
    wandb.init(project="Classification_Bert_GPT", name=run_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NEPOCH = 100
    BestEpoch=0
    BestLoss = np.Inf
    data = pickle.load(open("../data/data_classification.pk", "rb"))
    train_data_loader = Dataloader(seq_pos=data['train']['pos'], seq_neg=data['train']['neg'], batch_size=256)
    test_data_loader = Dataloader(seq_pos=data['test']['pos'], seq_neg=data['test']['neg'], batch_size=256,
                                  shuffle=False)
    model = BERT().to(device)
    optimizer = optim.AdamW(model.parameters(), fused=True)
    optimizer.zero_grad()
    loss_func = nn.BCEWithLogitsLoss()

    for epoch in range(NEPOCH):
        epoch_loss = 0
        for s, batch in enumerate(tqdm(train_data_loader)):
            padded_seq, batch_tar = batch
            padded_seq = padded_seq.to(device)
            batch_tar = batch_tar.to(device)
            src_mask = create_src_mask(padded_seq).to(device)
            outputs = model(padded_seq, src_mask)
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
                padded_seq, batch_tar = batch
                padded_seq = padded_seq.to(device)
                batch_tar = batch_tar.to(device)
                src_mask = create_src_mask(padded_seq).to(device)
                outputs = model(padded_seq, src_mask)
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
