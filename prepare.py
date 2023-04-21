"""a quick way to tokenize 2 languages"""
import pickle
import tiktoken
from tqdm import tqdm
enc = tiktoken.get_encoding("cl100k_base")

with open("../data/chinese.zh") as f:
    chinese_lines = f.readlines()
with open("../data/english.en") as f:
    english_lines = f.readlines()

"""first tokenize with tiktoken, and then encode that with a dict"""
def get_token_map(lines):
    token_map = {}
    tokenized_lines = []
    count = 1  # 0 is reserved for padding
    for line in tqdm(lines):
        line = line.strip()
        token_list = enc.encode(line)
        for token in token_list:
            if token not in token_map:
                token_map[token] = count
                count += 1
        tokenized_lines.append([token_map[x] for x in token_list])
    print("vocab size:", count)
    return token_map, tokenized_lines

chinese_token_map, chinese_tokenized_lines = get_token_map(chinese_lines)
english_token_map, english_tokenized_lines = get_token_map(english_lines)

pickle.dump([chinese_token_map, english_token_map], open("../data/token_map.pk", "wb"))
pickle.dump(chinese_tokenized_lines, open("../data/chinese_lines.pk", "wb"))
pickle.dump(english_tokenized_lines, open("../data/english_lines.pk", "wb"))
