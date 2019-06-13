# coding: utf-8
"""
    concatenate documents in one single file
"""

import argparse
import re
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from create_db import Sentence, Document

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset name")
    args = parser.parse_args()

    dataset = args.dataset

    # Load raw text of NIPS/ACL/VLDB dataset

    dataset_dir = Path('../data') / dataset
    engine = create_engine('sqlite:///{}/corpus.db'.format(dataset_dir.absolute()))
    DBSession = sessionmaker(bind=engine)

    print('Loading all sentences...')
    session = DBSession()
    sentences = session.query(Sentence).all()
    session.close()

    output_file = dataset_dir / 'docs_in_sent.txt'
    with open(output_file.absolute(), 'w') as fout:
        for sent in sentences:
            fout.write(sent.rawtext)
            fout.write('\n')

    del sentences

    print('Loading all documents...')
    session = DBSession()
    docs = session.query(Document).all()
    session.close()

    output_file = dataset_dir / 'docs_in_doc.txt'
    with open(output_file.absolute(), 'w') as fout:
        for doc in docs:
            rawtext = doc.rawtext
            rawtext = re.sub(r"([.,!:?()])", r" \1 ", rawtext)
            rawtext = re.sub(r"\s{2,}", " ", rawtext)
            fout.write(rawtext)
            fout.write('\n')
