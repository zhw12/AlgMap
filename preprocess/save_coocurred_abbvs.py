# coding: utf-8
"""
    Baseline methods: Co-occurred acronyms in sentence/document
"""

import argparse
import re
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from create_db import Document, Sentence

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset name")
    args = parser.parse_args()

    # Load raw text of NIPS/ACL/VLDB dataset
    dataset = args.dataset

    dataset_rootdir = Path('../data') / dataset
    engine = create_engine('sqlite:///{}/corpus.db'.format(dataset_rootdir.absolute()))
    DBSession = sessionmaker(bind=engine)
    print(engine)

    # load all sentences from db
    print('Loading all docs...')
    session = DBSession()
    docs = session.query(Document).all()
    session.close()

    pattern = r'\b(?:[a-z]*[A-Z][a-z\d+-]*){2,}\b'

    with (dataset_rootdir / 'docabbv.txt').open(mode='w') as fout:
        for doc in docs:
            abbvs = re.findall(pattern, doc.rawtext)
            fout.write(' '.join(abbvs))
            fout.write('\n')

    del docs

    session = DBSession()
    sentences = session.query(Sentence).all()
    session.close()

    print('Loading all sentences...')
    with (dataset_rootdir / 'sentabbv.txt').open(mode='w') as fout:
        for sent in sentences:
            abbvs = re.findall(pattern, sent.rawtext)
            fout.write(' '.join(abbvs))
            fout.write('\n')
