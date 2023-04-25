"""Simple Bert classification model, with only [CLS] token as first token"""
"""Simple GPT classification model, exactly the same as BERT, but with left-to-right mask"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import sys
import pickle
from tqdm import tqdm
import wandb
from transformer import create_src_mask, create_tgt_mask
from utils import BertorGPTDataloader, BERTorGPT

MODEL2TRAIN = sys.argv[1] # bert or gpt
assert MODEL2TRAIN in ['bert', 'gpt']

if __name__ == '__main__':
    run_name = f"{MODEL2TRAIN} classification"
    wandb.init(project="Classification_Bert_GPT", name=run_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NEPOCH = 100
    BestEpoch=0
    BestLoss = np.Inf
    data = pickle.load(open("../data/data_classification.pk", "rb"))
    train_data_loader = BertorGPTDataloader(seq_src=data['train']['src'], seq_pos=data['train']['pos'],
                                   seq_neg=data['train']['neg'], batch_size=256)
    test_data_loader = BertorGPTDataloader(seq_src=data['test']['src'], seq_pos=data['test']['pos'],
                                  seq_neg=data['test']['neg'], batch_size=256, shuffle=False)
    model = BERTorGPT().to(device)
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
            src_mask = create_src_mask(seq).to(device) if MODEL2TRAIN == 'bert' else create_tgt_mask(seq).to(device)
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
