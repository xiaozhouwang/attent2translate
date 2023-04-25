# Fun with Attention

## Seq2Seq
revisit RNN with attention, and vanilla transformer for chinese-english translation
#### see the experiments [here](https://wandb.ai/xiaozhou/Seq2Seq/workspace?workspace=user-xiaozhou)

## Classification with Bert and GPT
classify if it is a correct english-chinese pair with Bert and GPT. 

Note that this is a toy example that is not a natural fit for Bert or GPT. So neither Bert or GPT is "properly" done.

Anyway, for Bert, 

1. pretrain a classification model with randomly replacing the correct translation 50% of the time, as negative examples
2. no fine tuning. Just test on the hold out set that also has randomly replaced translation sentences.

for GPT,
1. for training, also randomly replacing the correct translation 50% of the time, and put the label (True/False) as the last token.
2. same test set as Bert, use the predicted last token as predicted label.

#### see the experiments [here](https://wandb.ai/xiaozhou/Classification_Bert_GPT/workspace?workspace=user-xiaozhou)
Note: with this setup, it seems it suffers from "curse of randomness", i.e. the performance can be drastically different without fixing the random seed.