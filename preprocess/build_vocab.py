# coding: utf-8
"""
    Build vocabulary (word, character) for dataset
"""

import argparse
import re
import sys
import ujson as json
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).absolute().parent.parent))
from common.util import get_line_count

exact_pattern = r'^(?:[a-z]*[A-Z][a-z\d+-]*){2,}$'


def trim_rule(word, count, min_count):
    """keep all acronyms and frequent words"""
    if re.search(exact_pattern, word):
        return True
    elif count >= min_count:
        return True
    else:
        return False


def word_tokenize(sent):
    tokens = sent.split()
    return tokens


def make_dict(counter, special_tokens, limit=5, trim_rule=None):
    token2idx_dict = {}
    for i, token in enumerate(special_tokens):
        token2idx_dict[token] = i

    index = len(special_tokens)
    for token in counter.keys():
        if trim_rule is not None:
            if trim_rule(token, counter[token], limit):
                token2idx_dict[token] = index
                index += 1
        else:
            if counter[token] >= limit:
                token2idx_dict[token] = index
                index += 1
    return token2idx_dict


def count_words(text_file, special_tokens):
    word_counter = Counter()
    char_counter = Counter()
    print("Counting words...")
    total = get_line_count(text_file)
    with open(text_file) as fin:
        for line in tqdm(fin, total=total):
            for token in word_tokenize(line):
                word_counter[token] += 1
                for char in token:
                    char_counter[char] += 1
    for token in special_tokens:
        del word_counter[token]
    return word_counter, char_counter


def get_embedding(token2idx, data_type, emb_file=None, vec_size=None):
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}
    if emb_file is not None:
        with open(emb_file) as fin:
            size, vec_size = fin.readline().strip().split()
            size, vec_size = int(size), int(vec_size)
            for line in tqdm(fin, total=size):
                array = line.split()
                # word = "".join(array[0:-vec_size])
                word = array[0]
                vector = list(map(float, array[-vec_size:]))
                if word in token2idx:
                    embedding_dict[word] = vector
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(token2idx), data_type))
    else:
        assert vec_size is not None
        for token in token2idx:
            embedding_dict[token] = [np.random.normal(
                scale=0.01) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(
            len(token2idx)))

    pad = "<PAD>"
    unk = "<UNK>"
    sep = "<SEP>"
    cls = "<CLS>"

    embedding_dict[pad] = [0. for _ in range(vec_size)]
    embedding_dict[unk] = [0. for _ in range(vec_size)]
    embedding_dict[sep] = [0. for _ in range(vec_size)]
    embedding_dict[cls] = [np.random.normal(scale=0.01) for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict.get(token, [np.random.normal(scale=0.01) for _ in range(vec_size)])
                    for token, idx in token2idx.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help="dataset name")
    parser.add_argument('--use_word_cnter', dest='use_word_cnter', action='store_true')
    parser.add_argument('--no-use_word_cnter', dest='use_word_cnter', action='store_false')
    parser.set_defaults(use_word_cnter=False)
    parser.add_argument('--limit', dest='limit', type=int, default=5)

    args = parser.parse_args()

    dataset = args.dataset
    limit = args.limit
    use_word_cnter = args.use_word_cnter

    dataset_dir = Path('../data') / dataset
    docs_in_sent_file = dataset_dir / 'docs_in_sent.txt'
    special_tokens = ['<PAD>', '<UNK>', '<SEP>', '<CLS>']

    if not use_word_cnter:
        word_counter, char_counter = count_words(str(docs_in_sent_file), special_tokens=special_tokens)
        with (dataset_dir / 'word_cnter.json').open('w') as fout:
            json.dump(word_counter, fout)

        with (dataset_dir / 'char_cnter.json').open('w') as fout:
            json.dump(char_counter, fout)
    else:
        with (dataset_dir / 'word_cnter.json').open() as fin:
            word_counter = json.load(fin)

        with (dataset_dir / 'char_cnter.json').open() as fin:
            char_counter = json.load(fin)

    word2ind = make_dict(word_counter, special_tokens=special_tokens, limit=limit,
                         trim_rule=trim_rule)
    char2ind = make_dict(char_counter, special_tokens=special_tokens, limit=limit,
                         trim_rule=trim_rule)

    abbv2ind = {}
    for word in word2ind:
        if re.search(exact_pattern, word):
            abbv2ind[word] = word2ind[word]

    word_emb_file = str(dataset_dir / 'word2vec.txt')
    word_emb = get_embedding(word2ind, "word", emb_file=word_emb_file)

    char_emb_dim = 50
    char_emb = get_embedding(char2ind, "char", vec_size=char_emb_dim)

    with (dataset_dir / 'word2id.json').open('w') as fout:
        json.dump(word2ind, fout)

    with (dataset_dir / 'char2id.json').open('w') as fout:
        json.dump(char2ind, fout)

    with (dataset_dir / 'abbv2ind.json').open('w') as fout:
        json.dump(abbv2ind, fout)

    with (dataset_dir / 'word_emb.json').open('w') as fout:
        json.dump(word_emb, fout)
