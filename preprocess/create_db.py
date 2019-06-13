# coding: utf-8
"""
    Store documents and sentences in database
"""

import argparse
import re
from pathlib import Path

import numpy as np
import spacy
from sqlalchemy import Column, String, Integer, Text, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])
nlp.add_pipe(nlp.create_pipe('sentencizer'))


def default(o):
    if isinstance(o, np.int64): return int(o)
    raise TypeError


# Initialize Database Schema
# Use sqlite3 with sqlalchemy

Base = declarative_base()


class Document(Base):
    __tablename__ = 'document'
    id = Column(Integer(), primary_key=True)
    filename = Column(String(20))
    dataset = Column(String(20))
    rawtext = Column(Text())
    year = Column(Integer())

    def __repr__(self):
        return (self.filename)


class Sentence(Base):
    __tablename__ = 'sentence'
    id = Column(Integer(), primary_key=True)
    rawtext = Column(Text())
    parent = Column(Integer(), ForeignKey('document.id'))

    def __repr__(self):
        return (self.rawtext)


# class Acronym(Base):
#     __tablename__ = 'acronym'
#     id = Column(Integer(), primary_key=True)
#     rawtext = Column(String(20))
#     fullname = Column(String(100), nullable=True)
#     abbvtype = Column(String(20))
#     parent = Column(Integer(), ForeignKey('sentence.id'))
#
#     def __repr__(self):
#         return (self.rawtext)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset name")
    args = parser.parse_args()

    dataset = args.dataset

    save_sent_and_doc = True
    save_doc_in_sent = False

    # Load raw text of NIPS/ACL/VLDB dataset
    txt_rootdir = Path('../data/PaperCrawler') / dataset / (dataset + 'papertxt' + 'Clean')
    txt_in_sent_rootdir = Path('../data/PaperCrawler') / dataset / (dataset + 'papertxt' + 'Sent')

    output_dir = Path('../data') / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    # use nltk to tokenize
    # sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    # now use spacy to tokenize

    engine = create_engine('sqlite:///{}/corpus.db'.format(output_dir.absolute()))
    DBSession = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)
    print(engine)

    if save_sent_and_doc:
        # Store sentences and documents in db
        docid = 0
        sentid = 0
        dataset_yearlist = sorted(txt_rootdir.iterdir(), reverse=True)
        for yeardir in tqdm(dataset_yearlist):
            if save_doc_in_sent:
                txt_in_sent_yeardir = txt_in_sent_rootdir / yeardir.name
                txt_in_sent_yeardir.mkdir(parents=True, exist_ok=True)
            session = DBSession()
            for txtfile in tqdm(sorted(list(yeardir.iterdir())), leave=False):
                with open(txtfile.absolute()) as f:
                    rawtext = f.read()
                    rawtext = re.sub(r'[^\x00-\x7F]+', '', rawtext)
                year = int(yeardir.name[-4:])
                new_document = Document(id=docid, filename=txtfile.name, rawtext=rawtext, year=year)
                sentences = []
                # for paragraph in rawtext.split('\n\n'):
                #     for sub_paragraph in paragraph.split('\n'):
                #         sentences.extend(sent_tokenizer.tokenize(sub_paragraph))
                for paragraph in rawtext.split('\n'):
                    paragraph = re.sub(r'\s+', ' ', paragraph)
                    paragraph = paragraph.strip()
                    if paragraph:
                        para = nlp(paragraph)
                        for sent in para.sents:
                            sentences.append(sent.text)

                for sent in sentences:
                    sent = re.sub(r'([.,!:?()])', r' \1 ', sent)
                    sent = re.sub(r'\s{2,}', ' ', sent)
                    new_sentence = Sentence(id=sentid, rawtext=sent, parent=new_document.id)
                    sentid += 1
                    session.add(new_sentence)
                docid += 1

                if save_doc_in_sent:
                    with (txt_in_sent_yeardir / txtfile.name).open(mode='w') as fout:
                        fout.write('\n'.join(sentences))

                session.add(new_document)
            session.commit()
            session.close()

    # if save_acro:
    #     # Store acronyms with details in db
    #     acronymid = 0
    #     acro_file = Path('../data') / dataset / 'acro.txt'
    #     acro_lines = [l for l in acro_file.open()]
    #     session = DBSession()
    #     for i in range(0, len(acro_lines), 4):
    #         abbv_line = acro_lines[i]
    #         sentid, _, _, abbv = abbv_line.strip().split(' ', 3)
    #         detail_line = acro_lines[i + 1]
    #         _, _, _, fullname = detail_line.strip().split(' ', 3)
    #         sentence_line = acro_lines[i + 2]
    #         new_acronym = Acronym(id=acronymid, rawtext=abbv, abbvtype='acro', fullname=fullname, parent=sentid)
    #         session.add(new_acronym)
    #         acronymid += 1
    #     session.commit()
    #     session.close()

    # '''
    #     Store abbreviations with pattern in db
    # '''
    #
    # session = DBSession()
    # with open('../data/NIPSabbv.txt', 'w') as fout:
    #     for i in tqdm(range(len(session.query(Document).all()))):
    #         acros = session.query(Acronym).join(Sentence).join(Document).filter(Document.id == i).filter(
    #             Acronym.abbvtype == 'abbv')
    #         fout.write(' '.join([a.rawtext for a in acros.all()]))
    #         fout.write('\n')
    # session.close()
