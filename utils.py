import pickle
import random
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

