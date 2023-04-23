import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle
from tqdm import tqdm
import wandb
from utils import get_train_test_split, S2SDataLoader
from seq2seq import Seq2Seq


if __name__ == '__main__':
    run_name = "multi head attention"
    wandb.init(project="Seq2Seq", name=run_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NEPOCH = 100
    BestEpoch=0
    BestLoss = np.Inf
    train_english, train_chinese, test_english, test_chinese = get_train_test_split()
    train_data_loader = S2SDataLoader(seq_chinese=train_chinese, seq_english=train_english, batch_size=128)
    test_data_loader = S2SDataLoader(seq_chinese=test_chinese, seq_english=test_english, batch_size=128, shuffle=False)
    model = Seq2Seq(device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), fused=True)
    optimizer.zero_grad()

    for epoch in range(NEPOCH):
        epoch_loss = 0
        for s, batch in enumerate(tqdm(train_data_loader)):
            padded_english, length_english, padded_chinese, length_chinese = batch
            padded_english = padded_english.to(device)
            padded_chinese = padded_chinese.to(device)
            outputs, loss = model(padded_english, length_english, padded_chinese)
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
                outputs, loss = model(padded_english, length_english, padded_chinese)
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
