# coding: utf-8
"""
    Parse acronyms in sentences
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from create_db import Sentence

sys.path.append(str(Path(__file__).absolute().parent.parent))
from common.util import enumerate_all_pairs
from models.util import load_vocab


# overlap
def overlap(pos1, pos2):
    '''
        assume entity1 comes first
    '''
    if pos1[1] >= pos2[0]:
        return True
    else:
        return False


# span len
def span(pos):
    return pos[1] - pos[0]


# get minimal span
def get_minimal_span_pairs(pos_pairs):
    # should assume pos1 < pos2
    #     pos_pairs = [(8, 11), (8, 14), (62, 69), (62, 71)]
    sorted_pos_pair = sorted(pos_pairs, key=lambda x: (x[0], x[1]))

    drop_idx = set()
    for i in range(0, len(sorted_pos_pair) - 1):
        if i in drop_idx:
            continue
        pos1 = sorted_pos_pair[i]
        j = i + 1
        while j < len(sorted_pos_pair):
            pos2 = sorted_pos_pair[j]
            if overlap(pos1, pos2):
                if span(pos1) > span(pos2):
                    drop_idx.add(i)
                    break
                elif span(pos1) < span(pos2):
                    drop_idx.add(j)
            else:
                break
            j += 1

    keep_pairs = []
    for i in range(0, len(sorted_pos_pair)):
        if i not in drop_idx:
            keep_pairs.append(sorted_pos_pair[i])
    return keep_pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help="dataset name")
    parser.add_argument('--limit', dest='limit', type=int, default=None)

    parser.add_argument('--cross_sentence_window_size', dest='cross_sentence_window_size', type=int, default=10)
    parser.add_argument('--max_para_len', dest='max_para_len', type=int, default=21)
    parser.add_argument('--max_sent_len', dest='max_sent_len', type=int, default=160)

    parser.add_argument('--add_sep', dest='add_sep', action='store_true')
    parser.add_argument('--no-add_sep', dest='add_sep', action='store_false')
    parser.set_defaults(add_sep=True)
    parser.add_argument('--add_cls', dest='add_cls', action='store_true')
    parser.add_argument('--no-add_cls', dest='add_cls', action='store_false')
    parser.set_defaults(add_cls=True)

    args = parser.parse_args()

    dataset = args.dataset
    limit = args.limit
    add_sep = args.add_sep
    add_cls = args.add_cls

    # cross_sentence_window_size = 10  # window for paragraph
    # max_para_len = 10 * 2 + 1
    # max_sent_len = 160
    cross_sentence_window_size = args.cross_sentence_window_size
    max_para_len = args.max_para_len
    max_sent_len = args.max_sent_len

    dataset_rootdir = Path('../data') / dataset
    engine = create_engine('sqlite:///{}/corpus.db'.format(dataset_rootdir.absolute()))
    DBSession = sessionmaker(bind=engine)
    print(engine)

    # exact_pattern = '^[A-Z][\w\+\-]*[A-Z]+[\w\+\-]*$'
    # pattern = '(?:[a-z]*[A-Z][a-z]*){2,}'
    exact_pattern = r'^(?:[a-z]*[A-Z][a-z\d+-]*){2,}$'

    # load all sentences from db
    print('Loading all sentences...')
    session = DBSession()
    if limit is None:
        sentences = session.query(Sentence).all()
    else:
        sentences = session.query(Sentence).limit(limit).all()
    session.close()

    docid2abbvsents = defaultdict(list)


    def sentid2tokens(id_range):
        if isinstance(id_range, int):
            i = id_range
            sent = sentences[i]
            return sent.rawtext.strip().split()
        sentence_tokens = []
        for i in id_range:
            sent = sentences[i]
            sentence_tokens.append(sent.rawtext.strip().split())
        return sentence_tokens


    # extract candidate sentences that will consider relations
    print('calculating inner sentence pairs ...')
    inner_sentence_pairs = defaultdict(list)
    for sent in tqdm(sentences):
        rawtext = sent.rawtext.strip()
        tokens = rawtext.split()
        abbvs = []
        for i, token in enumerate(tokens):
            if re.match(exact_pattern, token):
                abbvs.append({'token': token, 'pos': i})
        if len(abbvs) > 0:
            docid2abbvsents[sent.parent].append({'sent': sent, 'abbvs': abbvs})
        if len(abbvs) >= 2:
            cand_pairs = enumerate_all_pairs(abbvs)
            for cand_pair in cand_pairs:
                e1 = cand_pair[0]['token']
                e2 = cand_pair[1]['token']
                pos1 = cand_pair[0]['pos']
                pos2 = cand_pair[1]['pos']
                if e1 != e2:
                    e1, e2 = sorted([e1, e2])  # make sure the form of entity pair is unique
                    inner_sentence_pairs[e1, e2].append([sent.id, pos1, pos2])

    print('calculating cross sentence pairs ...')
    cross_sentence_pairs = defaultdict(list)
    for docid, abbvsents_in_doc in tqdm(docid2abbvsents.items()):
        cross_sentence_pairs_in_doc = defaultdict(list)

        num_abbvsents_in_doc = len(abbvsents_in_doc)
        for i in range(0, num_abbvsents_in_doc - 1):
            abbvsent1 = abbvsents_in_doc[i]
            abbvs1 = [abbv['token'] for abbv in abbvsents_in_doc[i]['abbvs']]
            # for j in range(i + 1, num_abbvsents_in_doc):
            for j in range(i, num_abbvsents_in_doc):
                abbvsent2 = abbvsents_in_doc[j]
                if abs(abbvsent1['sent'].id - abbvsent2['sent'].id) > cross_sentence_window_size:
                    break
                abbvs2 = [abbv['token'] for abbv in abbvsents_in_doc[j]['abbvs']]
                for abbv1 in abbvs1:
                    for abbv2 in abbvs2:
                        if abbv1 != abbv2 and abbvsent1['sent'].id != abbvsent2['sent'].id:
                            # if abbv1 != abbv2:
                            # if abbvsent1['sent'].id == abbvsent2['sent'].id:
                            #     import pdb; pdb.set_trace();
                            abbv1, abbv2 = sorted([abbv1, abbv2])
                            cross_sentence_pairs_in_doc[abbv1, abbv2].append(
                                tuple(((abbvsent1['sent'].id, abbvsent2['sent'].id))))  # should assume pos1 < pos2
        for key, values in cross_sentence_pairs_in_doc.items():
            values = list(set(values))
            cross_sentence_pairs[key].extend(get_minimal_span_pairs(values))
    del docid2abbvsents

    word2ind_file = dataset_rootdir / 'word2id.json'
    word2ind, vocab = load_vocab(str(word2ind_file))
    unk_idx = word2ind['<UNK>']

    # inner sentence pairs data
    # Do not cut off long sentences

    # inner sentence pairs for pcnn does not insert speical tokens
    print('calculating data (indexes) for inner sentence pairs ...')
    inner_sentence_pairs_data = defaultdict(list)
    for e1, e2 in tqdm(inner_sentence_pairs):
        for d in inner_sentence_pairs[e1, e2]:
            sentid = d[0]
            pos1 = d[1]
            pos2 = d[2]
            tokens = sentid2tokens(sentid)
            inds = [word2ind.get(w, unk_idx) for w in tokens]
            if add_cls:
                inds.insert(0, word2ind['<CLS>'])
                pos1 += 1
                pos2 += 1
            if add_sep:
                inds.append(word2ind['<SEP>'])
            inner_sentence_pairs_data[e1, e2].append([pos1, pos2] + inds)

    with (dataset_rootdir / 'inner_sentence_pairs_data.json').open(mode='w') as fout:
        for key, value in inner_sentence_pairs_data.items():
            fout.write(json.dumps([key, value]))
            fout.write('\n')

    # cross sentence pairs data
    # throw away long paragraphs?
    print('calculating data (indexes) for cross sentence pairs ...')
    cross_sentence_pairs_data = defaultdict(list)
    for e1, e2 in tqdm(cross_sentence_pairs):
        for st, ed in cross_sentence_pairs[e1, e2]:
            paragraph_inds = []
            sentence_tokens = sentid2tokens(range(st, ed + 1))
            assert len(sentence_tokens) <= max_para_len

            for tokens in sentence_tokens:
                inds = [word2ind.get(w, unk_idx) for w in tokens]
                # inds = inds[:max_sent_len]
                paragraph_inds.append(inds)
            cross_sentence_pairs_data[e1, e2].append(paragraph_inds)

    with (dataset_rootdir / 'cross_sentence_pairs_data.json').open(mode='w') as fout:
        for key, value in cross_sentence_pairs_data.items():
            fout.write(json.dumps([key, value]))
            fout.write('\n')

    # paragraph pairs data for pcnn_paragraph
    # do not cut off long paragraphs and sentences
    print('calculating data (indexes) for paragraph pairs ...')
    # paragraph_pairs_data = defaultdict(list)
    paragraph_pairs_data = defaultdict(list, inner_sentence_pairs_data)
    del inner_sentence_pairs_data
    for e1, e2 in tqdm(cross_sentence_pairs):
        for st, ed in cross_sentence_pairs[e1, e2]:
            sentence_tokens = sentid2tokens(range(st, ed + 1))
            paragraph_tokens = []
            if add_cls:
                paragraph_tokens.append('<CLS>')
            for sent_tokens in sentence_tokens:
                paragraph_tokens.extend(sent_tokens)
                if add_sep:
                    paragraph_tokens.append('<SEP>')
            # paragraph_tokens = paragraph_tokens[:-1]

            # find positions after all special tokens are inserted
            for i, token in enumerate(paragraph_tokens):
                if e1 == token:
                    pos1 = i
                elif e2 == token:
                    pos2 = i
            # swap when pos1 > pos2
            if pos1 > pos2:
                pos1, pos2 = pos2, pos1

            inds = [word2ind.get(w, unk_idx) for w in paragraph_tokens]

            paragraph_pairs_data[e1, e2].append([pos1, pos2] + inds)

    with (dataset_rootdir / 'paragraph_pairs_data.json').open(mode='w') as fout:
        for key, value in paragraph_pairs_data.items():
            fout.write(json.dumps([key, value]))
            fout.write('\n')
