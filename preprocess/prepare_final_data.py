# coding: utf-8
"""
    Cut off long paragraphs and long sentences
    Do labelling, padding, and data splitting
"""

import argparse
import re
import sys
import ujson as json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append(str(Path(__file__).absolute().parent.parent))
from common.util import enumerate_all_pairs, get_line_count, list_to_sparse_dict
from models.util import load_vocab
from itertools import islice


def is_abbv(token, exact_pattern=r'^(?:[a-z]*[A-Z][a-z\d+-]*){2,}$'):
    return re.match(exact_pattern, token)


def get_char_ids(token, char2ind, special_tokens=['<PAD>', '<UNK>', '<SEP>', '<CLS>']):
    unk_idx = char2ind['<UNK>']
    if token not in special_tokens:
        return [char2ind.get(char, unk_idx) for char in token]
    else:
        return [char2ind[token]]


def get_pos_feature(sent_len, epos, limit, max_len):
    '''
    clip the postion range:
    : -limit ~ limit => 0 ~ limit * 2 + 2
    : -51 => 1
    : -50 => 1
    : 50 => 101
    : >50: 102
    '''

    def padding(x):
        if x < 1:
            return 1
        elif x > limit * 2 + 1:
            return limit * 2 + 1
        else:
            return int(x)

    if sent_len < max_len:
        index = np.arange(sent_len)
    else:
        index = np.arange(max_len)

    pf1 = list(map(padding, index - epos[0] + 2 + limit))
    pf2 = list(map(padding, index - epos[1] + 2 + limit))

    if len(pf1) < max_len:
        pf1 += [0] * (max_len - len(pf1))
        pf2 += [0] * (max_len - len(pf2))

    pf1[0] = 0
    pf2[0] = 0

    return [pf1, pf2]


def get_pad_sentence(sent, max_len, pad_ind=0):
    if len(sent) < max_len:
        sent += [pad_ind] * (max_len - len(sent))
        return sent
    else:
        return sent[:max_len]


def save_bags_feature_by_id(all_pairs, pair_ids, pairs_data, pair2label, word2ind, char2ind, vocab, filepath, limit,
                            max_len, max_char_len=30, use_sparse=False):
    """
        example = {"entity_ids": entity_ids, "num_instances": num_instances,
                   "paragraph_ids": paragraph_ids, "position_features": position_features,
                   "entity_positions": entity_positions, "num_abbreviations": num_abbreviations,
                   "abbbreviation_ids": abbbreviation_ids, "abbreviation_match": abbreviation_match,
                   "entity_char_ids": entity_char_ids, "paragraph_char_ids": paragraph_char_ids,
                   "y": y}
    """
    pad_idx = word2ind['<PAD>']
    unk_idx = word2ind['<UNK>']
    cls_idx = word2ind['<CLS>']
    sep_idx = word2ind['<SEP>']

    special_ids = [pad_idx, unk_idx, cls_idx, sep_idx]

    fout = open(filepath, 'w')
    # bags_feature = []
    for pair_id in tqdm(pair_ids):
        e1, e2 = all_pairs[pair_id]
        eid1, eid2 = word2ind.get(e1, unk_idx), word2ind.get(e2, unk_idx)
        y = pair2label[e1, e2]

        entity_ids = [eid1, eid2]
        paragraph_ids = pairs_data[e1, e2]
        num_instances = len(paragraph_ids)

        valid_paragraph_ids = []
        entity_positions = []
        position_features = []
        for i in range(num_instances):
            instance_inds = paragraph_ids[i]
            epos = instance_inds[0], instance_inds[1]
            if epos[0] >= max_len - 1 or epos[1] >= max_len - 1:
                # cutoff sentenfecs with overflow entities
                continue
            else:
                sent_ids = instance_inds[2:]
                entity_positions.append(instance_inds[:2])

                pad_sent_ids = get_pad_sentence(sent_ids, max_len=max_len, pad_ind=pad_idx)
                pf = get_pos_feature(len(sent_ids), instance_inds[:2], limit=limit, max_len=max_len)

                valid_paragraph_ids.append(pad_sent_ids)
                position_features.append(pf)

        paragraph_ids = valid_paragraph_ids
        num_instances = len(valid_paragraph_ids)

        num_abbreviations = []
        abbbreviation_ids = []
        abbreviation_positions = []
        for instance_inds in paragraph_ids:
            tokens = [vocab[id] for id in instance_inds]
            abbv_ids = []
            abbv_pos = []
            for i, d in enumerate(zip(tokens, instance_inds)):
                token, id = d
                if is_abbv(token):
                    abbv_ids.append(id)
                    abbv_pos.append(i)

            num_abbreviations.append(len(abbv_ids))
            abbbreviation_ids.append(abbv_ids)
            abbreviation_positions.append(abbv_pos)

        if num_instances == 0:
            sent_ids = [cls_idx, sep_idx]
            position_features = [[get_pad_sentence([0], max_len=max_len), get_pad_sentence([0], max_len=max_len)]]
            paragraph_ids = [get_pad_sentence(sent_ids, max_len=max_len)]
            entity_positions = [[0, 1]]
            num_abbreviations = [0]
            abbbreviation_ids = [entity_ids]
            abbreviation_positions = entity_positions
            num_instances = 1  # add a pseudo instance

        entity_char_ids = [get_char_ids(e, char2ind) for e in [e1, e2]]
        entity_char_ids = [get_pad_sentence(char_ids, max_len=max_char_len, pad_ind=pad_idx) for char_ids in
                           entity_char_ids]

        paragraph_char_ids = []
        for instance_inds in paragraph_ids:
            instance_char_ids = [get_char_ids(vocab[id], char2ind) for id in instance_inds]
            pad_instance_char_ids = [get_pad_sentence(char_ids, max_len=max_char_len, pad_ind=pad_idx) for char_ids in
                                     instance_char_ids]
            paragraph_char_ids.append(pad_instance_char_ids)

        abbreviation_match = []
        for abbv_pos in abbreviation_positions:
            abbv_match = [0] * max_len
            for id in abbv_pos:
                abbv_match[id] = 1
            abbreviation_match.append(abbv_match)

        if use_sparse:
            paragraph_ids = list_to_sparse_dict(paragraph_ids)
            abbreviation_match = list_to_sparse_dict(abbreviation_match)
            paragraph_char_ids = list_to_sparse_dict(paragraph_char_ids)
            position_features = list_to_sparse_dict(position_features)

        example = {"entity_ids": entity_ids, "num_instances": num_instances,
                   "paragraph_ids": paragraph_ids, "position_features": position_features,
                   "entity_positions": entity_positions, "num_abbreviations": num_abbreviations,
                   "abbbreviation_ids": abbbreviation_ids, "abbreviation_match": abbreviation_match,
                   "entity_char_ids": entity_char_ids, "paragraph_char_ids": paragraph_char_ids,
                   "y": y}
        fout.write(json.dumps(example))
        fout.write('\n')
    fout.close()
    # return bags_feature


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help="dataset name")
    parser.add_argument('mode', default='standard')

    args = parser.parse_args()

    dataset = args.dataset
    mode = args.mode

    assert mode == 'standard' or mode == 'paragraph'

    dataset_rootdir = Path('../data') / dataset
    # engine = create_engine('sqlite:///{}/corpus.db'.format(dataset_rootdir.absolute()))
    # DBSession = sessionmaker(bind=engine)
    # print(engine)

    # manual_entities_dict = {}
    # l2i = {'pos': 1, 'neg': 0}
    # manual_file = Path('../manual') / (dataset.lower() + '.json')
    # with manual_file.open() as fin:
    #     for line in fin:
    #         d = json.loads(line)
    #         _, e1, e2, _ = d['text'].split(' ', 3)
    #         l = l2i[d['labels'][0]]
    #         entities = tuple(sorted([e1, e2]))
    #         manual_entities_dict[entities] = l

    # max_para_len = 10 * 2 + 1
    if mode == 'paragraph':
        limit = 80
        max_sent_len = 160
        pairs_data_file = str(dataset_rootdir / 'paragraph_pairs_data.json')
        feature_dir = dataset_rootdir / 'paragraph'
    else:
        limit = 80
        max_sent_len = 160
        pairs_data_file = str(dataset_rootdir / 'inner_sentence_pairs_data.json')
        feature_dir = dataset_rootdir / 'standard'

    print('Loading {} pairs data...'.format(mode))

    all_pairs_file = dataset_rootdir / 'all_pairs.json'

    if not all_pairs_file.exists():
        all_pairs = []
        with (dataset_rootdir / 'inner_sentence_pairs_data.json').open() as fin:
            for line in fin:
                key, _ = json.loads(line)
                all_pairs.append(tuple(key))
        with (dataset_rootdir / 'paragraph_pairs_data.json').open() as fin:
            for line in fin:
                key, _ = json.loads(line)
                all_pairs.append(tuple(key))
        all_pairs = sorted(list(set(all_pairs)))

        with (dataset_rootdir / 'all_pairs.json').open(mode='w') as fout:
            for p in all_pairs:
                fout.write(json.dumps(p))
                fout.write('\n')

    all_pairs = []
    with all_pairs_file.open() as fin:
        for line in fin:
            entities = json.loads(line)
            if entities[0] != entities[1]:
                all_pairs.append(tuple(entities))

    pairs_data = {p: [] for p in all_pairs}

    total = get_line_count(pairs_data_file)
    with open(pairs_data_file) as fin:
        for s in tqdm(islice(fin, None), total=total):
            key, value = json.loads(s)
            # skip equal entities
            pairs_data[tuple(key)] = value

    # get co-occurrence statistics from table
    table_abbv_hist = Counter()
    co_occur = defaultdict(int)
    abbvfile = dataset_rootdir / 'table_align_clean.txt'
    with abbvfile.open() as fin:
        for line in fin:
            line = line.strip()
            if line:
                abbvs = line.split(' ')
                for w in abbvs:
                    table_abbv_hist[w] += 1
                pairs = enumerate_all_pairs(abbvs)
                for w1, w2 in pairs:
                    co_occur[(w1, w2)] += 1
                    co_occur[(w2, w1)] += 1

    # weak supervised labels with table co-occurrence
    pair2label = {}
    for e1, e2 in pairs_data:
        co_occur_num = co_occur[e1, e2]
        if co_occur_num > 0:
            pair2label[e1, e2] = 1
        else:
            pair2label[e1, e2] = 0

    pair_ids = range(len(all_pairs))
    pair2id = {pairs: i for i, pairs in enumerate(all_pairs)}
    # manual_pair_ids = set([pair2id[entities] for entities in manual_entities_dict if entities in pair2id])

    pair_ids_train_, pair_ids_test = train_test_split(pair_ids, test_size=0.2,
                                                      random_state=0)

    pair_ids_train, pair_ids_valid = train_test_split(pair_ids_train_,
                                                      test_size=0.1,
                                                      random_state=0)

    # pair_ids_train = [id for id in pair_ids_train if id not in manual_pair_ids]
    # pair_ids_valid = [id for id in pair_ids_valid if id not in manual_pair_ids]
    # pair_ids_test = [id for id in pair_ids_test if id not in manual_pair_ids]
    # pair_ids_test += list(manual_pair_ids)

    labels_train = [pair2label[all_pairs[id]] for id in pair_ids_train]
    labels_valid = [pair2label[all_pairs[id]] for id in pair_ids_valid]
    labels_test = [pair2label[all_pairs[id]] for id in pair_ids_test]

    word2ind_file = dataset_rootdir / 'word2id.json'
    word2ind, vocab = load_vocab(str(word2ind_file))

    char2ind_file = dataset_rootdir / 'char2id.json'
    char2ind, char_vocab = load_vocab(str(char2ind_file))

    (feature_dir / 'train').mkdir(parents=True, exist_ok=True)
    (feature_dir / 'valid').mkdir(parents=True, exist_ok=True)
    (feature_dir / 'test').mkdir(parents=True, exist_ok=True)
    (feature_dir / 'all').mkdir(parents=True, exist_ok=True)

    print('writing train/labels.txt ...')
    with open(feature_dir / 'train/labels.txt', 'w') as fout:
        for label in labels_train:
            fout.write(str(label))
            fout.write('\n')

    print('writing valid/labels.txt ...')
    with open(feature_dir / 'valid/labels.txt', 'w') as fout:
        for label in labels_valid:
            fout.write(str(label))
            fout.write('\n')

    print('writing test/labels.txt ...')
    with open(feature_dir / 'test/labels.txt', 'w') as fout:
        for label in labels_test:
            fout.write(str(label))
            fout.write('\n')

    with open(feature_dir / 'all/all_pairs.txt', 'w') as fout:
        for pair_id in tqdm(pair_ids_train):
            e1, e2 = all_pairs[pair_id]
            fout.write('{} {}\n'.format(e1, e2))
        for pair_id in tqdm(pair_ids_test):
            e1, e2 = all_pairs[pair_id]
            fout.write('{} {}\n'.format(e1, e2))

    print('calculating bags_feature_train ...')
    save_bags_feature_by_id(all_pairs, pair_ids_train, pairs_data, pair2label, word2ind, char2ind, vocab,
                            filepath=feature_dir / 'train/bags_feature.json', limit=limit, max_len=max_sent_len,
                            use_sparse=True)

    print('calculating bags_feature_valid ...')
    save_bags_feature_by_id(all_pairs, pair_ids_valid, pairs_data, pair2label, word2ind, char2ind, vocab,
                            filepath=feature_dir / 'valid/bags_feature.json', limit=limit, max_len=max_sent_len,
                            use_sparse=True)

    print('calculating bags_feature_test ...')
    save_bags_feature_by_id(all_pairs, pair_ids_test, pairs_data, pair2label, word2ind, char2ind, vocab,
                            filepath=feature_dir / 'test/bags_feature.json', limit=limit, max_len=max_sent_len,
                            use_sparse=True)
