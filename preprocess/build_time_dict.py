"""
    Record occurrence information of acronyms in the corpus
"""
import argparse
import sys

sys.path.append("../preprocess/")
from tqdm import tqdm
import re
from collections import Counter
from collections import defaultdict
from pathlib import Path
import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from create_db import Document

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help="dataset name")
    args = parser.parse_args()

    exact_pattern = r'^(?:[a-z]*[A-Z][a-z\d+-]*){2,}$'

    dataset_rootdir = Path('../data') / args.dataset
    engine = create_engine('sqlite:///{}/corpus.db'.format(dataset_rootdir.absolute()))
    DBSession = sessionmaker(bind=engine)
    print(engine)

    session = DBSession()
    docs = session.query(Document).all()
    session.close()

    time_cnter_dict = defaultdict(Counter)

    for doc in tqdm(docs):
        tokens = doc.rawtext.split()
        for token in tokens:
            if re.match(exact_pattern, token):
                time_cnter_dict[token][doc.year] += 1

    time_dict = Counter({w: min(time_cnter_dict[w]) for w in time_cnter_dict})

    with open(dataset_rootdir / 'time_cnter_dict.json', 'w') as fout:
        json.dump(time_cnter_dict, fout)
