"""prepare for classification with Bert and GPT"""
import pickle
import tiktoken
from tqdm import tqdm
import random
from utils import preprocess_sentences, get_train_test_split, create_negative_samples
enc = tiktoken.get_encoding("cl100k_base")
random.seed(124)

with open("../data/chinese.zh") as f:
    chinese_lines = f.readlines()
with open("../data/english.en") as f:
    english_lines = f.readlines()

"""first tokenize with tiktoken, and then encode that with a dict"""
def tokenize(english_lines, chinese_lines):
    tokenized_results = {'english': [], 'chinese': []}
    #BERT: 0 is reserved for padding, 1 is reserved for CLS token
    #GPT: 0 is reserved for padding, 1 is reserved for Positive, and 2 is reserved for Negative
    count = 3
    token_map = {}
    for k, lines in {'english': english_lines, 'chinese': chinese_lines}.items():
        for line in tqdm(lines):
            line = line.strip()
            token_list = enc.encode(line)
            for token in token_list:
                if token not in token_map:
                    token_map[token] = count
                    count += 1
            tokenized_results[k].append([token_map[x] for x in token_list])
    print("vocab size:", count)
    return tokenized_results, token_map


tokenized_results, token_map = tokenize(english_lines, chinese_lines)
new_english, new_chinese = preprocess_sentences(tokenized_results['english'],
                                               tokenized_results['chinese'])
train_english, positive_train_chinese, test_english, positive_test_chinese = get_train_test_split(new_chinese,
                                                                                                  new_english)
print("generating negative samples")
negative_train_chinese = create_negative_samples(positive_train_chinese)
negative_test_chinese = create_negative_samples(positive_test_chinese)

data = {'train': {'pos': positive_train_chinese,
                  'neg': negative_train_chinese,
                  'src': train_english},
        'test': {'pos': positive_test_chinese,
                 'neg': negative_test_chinese,
                 'src': test_english}
        }
print("saving data")
pickle.dump(data, open("../data/data_classification.pk", "wb"))



