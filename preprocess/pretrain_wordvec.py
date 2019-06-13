# coding: utf-8
"""
    Pretrain word vectors with domain corpus using gensim word2vec
"""

import argparse
import logging
import re
from pathlib import Path

from gensim.models.word2vec import Word2Vec, LineSentence
from gensim.utils import RULE_KEEP, RULE_DEFAULT

exact_pattern = r'^(?:[a-z]*[A-Z][a-z\d+-]*){2,}$'


def trim_rule(word, count, min_count):
    if re.search(exact_pattern, word):
        return RULE_KEEP
    elif count >= min_count:
        return RULE_DEFAULT


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset name")
    args = parser.parse_args()

    # NIPS/ACL/VLDB dataset
    dataset = args.dataset

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    dataset_rootdir = Path('../data/') / dataset
    txtfile_doc_per_line = dataset_rootdir / 'docs_in_doc.txt'
    wordvec_file = dataset_rootdir / 'word2vec.txt'
    vocab_file = dataset_rootdir / 'vocab.txt'

    model = Word2Vec(LineSentence(txtfile_doc_per_line.open(), limit=None), min_count=5, window=5, size=100,
                     max_vocab_size=None, trim_rule=trim_rule)
    model.wv.save_word2vec_format(str(wordvec_file.absolute()), binary=False)

    with vocab_file.open(mode='w') as fout:
        for v in model.wv.index2word:
            fout.write(v)
            fout.write('\n')
