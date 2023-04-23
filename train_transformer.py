import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle
from tqdm import tqdm
import wandb
from transformer import Transformer, create_src_mask, create_tgt_mask
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


def get_train_test_split():
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


class DataLoader:
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


if __name__ == '__main__':
    run_name = "vanilla transformer"
    wandb.init(project="Seq2Seq", name=run_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NEPOCH = 100
    BestEpoch=0
    BestLoss = np.Inf
    train_english, train_chinese, test_english, test_chinese = get_train_test_split()
    train_data_loader = DataLoader(seq_chinese=train_chinese, seq_english=train_english, batch_size=128)
    test_data_loader = DataLoader(seq_chinese=test_chinese, seq_english=test_english, batch_size=128, shuffle=False)
    model = Transformer().to(device)
    optimizer = optim.AdamW(model.parameters(), fused=True)
    optimizer.zero_grad()
    loss_func = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(NEPOCH):
        epoch_loss = 0
        for s, batch in enumerate(tqdm(train_data_loader)):
            padded_english, length_english, padded_chinese, length_chinese = batch
            padded_english = padded_english.to(device)
            padded_chinese = padded_chinese.to(device)
            input_target = padded_chinese[:, :-1]
            output_target = padded_chinese[:, 1:].contiguous().view(-1)
            src_mask = create_src_mask(padded_english).to(device)
            tgt_mask = create_tgt_mask(input_target, device=device).to(device)
            outputs = model(padded_english, input_target, src_mask, tgt_mask)

            output_dim = outputs.shape[-1]
            outputs = outputs.contiguous().view(-1, output_dim)

            loss = loss_func(outputs, output_target)
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
                padded_english, length_english, padded_chinese, length_chinese = batch
                padded_english = padded_english.to(device)
                padded_chinese = padded_chinese.to(device)
                input_target = padded_chinese[:, :-1]
                output_target = padded_chinese[:, 1:].contiguous().view(-1)
                src_mask = create_src_mask(padded_english).to(device)
                tgt_mask = create_tgt_mask(input_target, device=device).to(device)
                outputs = model(padded_english, input_target, src_mask, tgt_mask)

                output_dim = outputs.shape[-1]
                outputs = outputs.contiguous().view(-1, output_dim)

                loss = loss_func(outputs, output_target)
                epoch_loss += loss.item()
            epoch_loss /= len(test_data_loader)
            print(f"epoch: {epoch}, test loss: {epoch_loss}")
            wandb.log({'test_loss': epoch_loss, 'epoch': epoch})

        if epoch_loss < BestLoss:
            BestLoss = epoch_loss
            BestEpoch = epoch + 1
            print(f"saving best model with loss {BestLoss} at epoch {BestEpoch}")
            torch.save(model.state_dict(), f"../data/{run_name}.pt")

        if epoch - 3 > BestEpoch:
            print(f"early stop at {epoch+1} with best epoch {BestEpoch} and test loss {BestLoss}.")
            break
