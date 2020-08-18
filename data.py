import sys, csv
from collections import Counter

import numpy as np
import tensorflow.keras as kr


def read_file(filename):
    contents, labels = [], []
    with open(filename, 'r', encoding='utf-8') as f:
    #with open(filename, 'r') as f:
        for line in f:
            try:
                content, label = line.strip().split(',')
                if content:
                    contents.append(list(content))
                    labels.append(label)
            except:
                pass
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    print("Building vocab file...")
    data_train, _ = read_file(train_dir)
    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size-1)
    words, _ = list(zip(*count_pairs))
    words = ['<PAD>'] + list(words)
    with open(vocab_dir, 'w') as f:
        f.write('\n'.join(words) +'\n')

    print("Building vocab file completes")


def read_vocab(vocab_dir):
    with open(vocab_dir, 'r', encoding='utf-8') as f:
    # with open(vocab_dir, 'r') as f:
        words = [_.strip() for _ in f.readlines()]

    word2id = dict(zip(words, range(len(words))))

    return words, word2id


def read_category():
    categories = ['ART', 'PER', 'ORG']
    cat2id = dict(zip(categories, range(len(categories))))

    return categories, cat2id


def to_words(content, words):
    return ''.join(words[x] for x in content)


def process_file(filename, word2id, cat2id, max_length=25):
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word2id[x] for x in contents[i] if x in word2id])
        label_id.append(cat2id[labels[i]])

    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat2id))

    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    data_len = len(x)
    num_batch = data_len // batch_size + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i+1)*batch_size, data_len)
        yield x_shuffle[start_id: end_id], y_shuffle[start_id: end_id]


def batch_iter_v2(x, y, batch_size=64):
    data_len = len(x)
    num_batch = data_len // batch_size + 1

    # indices = np.random.permutation(np.arange(data_len))
    # x_shuffle = x[indices]
    # y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i+1)*batch_size, data_len)
        yield x[start_id: end_id], y[start_id: end_id]
      

'''extract distinct entities from the triplets file, becasue most entities will appear more than one time.'''
def extract_distinct_ett(file):
    entities = set()
    num = 0
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if '\0' in line:
                line.replace('\0','')
            if not line:
                continue
            entities.add(line.split(',')[0])
            num += 1
            if num % 1000000 == 0:
                print(len(entities))
    return entities


'''Write the content into the file'''
def write_list(elist, file):
    with open(file, 'w', encoding='utf8') as f:
        for e in elist:
            f.write(e+'\n')