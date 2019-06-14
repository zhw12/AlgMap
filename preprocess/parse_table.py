# coding: utf-8
"""
    Parse tablejson files and generate weakly supervised comparative pairs
"""

import argparse
import json
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from create_db import Document


def default(o):
    if isinstance(o, np.int64): return int(o)
    raise TypeError


def merge_cell(cell1, cell2):
    left1, up1, right1, down1 = cell1['TextBB']
    text1 = cell1['Text']
    left2, up2, right2, down2 = cell2['TextBB']
    text2 = cell2['Text']
    left = min(left1, left2)
    right = max(right1, right2)
    up = min(up1, up2)
    down = max(down1, down2)
    text = text1 + ' ' + text2
    return {'Rotation': 0, 'TextBB': [left, up, right, down], 'Text': text}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset name")
    args = parser.parse_args()

    # Load raw text of NIPS/ACL/VLDB dataset
    dataset = args.dataset

    dataset_dir = Path('../data') / dataset
    engine = create_engine('sqlite:///{}/corpus.db'.format(dataset_dir.absolute()))
    DBSession = sessionmaker(bind=engine)

    # Generate Positive Technology Candidates

    first_time = False

    valid_cell_texts = []

    rootdir = Path('../data/PaperCrawler')
    pdf_rootdir = rootdir / dataset / (dataset + 'paper')
    table_rootdir = rootdir / dataset / (dataset + 'paper' + 'tablejson')
    # pattern = '[A-Z][\w\+\-]*[A-Z]+[\w\+\-]*'
    # exact_pattern = '^[A-Z][\w\+\-]*[A-Z]+[\w\+\-]*$'
    pattern = r'(?:[a-z]*[A-Z][a-z\d+-]*){2,}'
    exact_pattern = r'^(?:[a-z]*[A-Z][a-z\d+-]*){2,}$'

    # extract all tables from paper
    # tables = defaultdict(list)
    tables = {}
    for yeardir in tqdm(sorted(table_rootdir.iterdir(), reverse=True), desc='year'):
        year = int(yeardir.name[-4:])
        # blackout years for table extractions
        if year <= 1992:
            continue
        for filename in tqdm(sorted(list(yeardir.iterdir())), leave=False, desc='file'):
            table_file = table_rootdir / yeardir.name / filename.name
            if table_file.name.endswith('.json'):
                file_prefix = table_file.name[:-5]
            else:
                file_prefix = table_file.name[:-4]
            with open(table_file) as fin:
                line = fin.read()
                table_json = json.loads(line)

            calib_table_json = []
            for t_json in table_json:
                if t_json['Type'] == 'Table':
                    calib_t_json = deepcopy(t_json)
                    ImageText = calib_t_json['ImageText']
                    ImageBB = calib_t_json['ImageBB']
                    left_bound, up_bound = ImageBB[0], ImageBB[1]
                    calib_ImageText = []
                    for cell in ImageText:
                        left, up, right, down = cell['TextBB']
                        cell['TextBB'] = left - left_bound, up - up_bound, right - left_bound, down - up_bound
                        cell['TextBB'] = np.round(cell['TextBB'], 3).tolist()
                        calib_ImageText.append(cell)
                    calib_t_json['ImageBB'] = [0, 0, ImageBB[2] - left_bound, ImageBB[3] - up_bound]
                    calib_t_json['ImageBB'] = np.round(calib_t_json['ImageBB'], 3).tolist()
                    calib_t_json['ImageText'] = calib_ImageText
                    calib_table_json.append(calib_t_json)

            for k, t_json in enumerate(calib_table_json):
                if t_json['Type'] == 'Table' and len(t_json['ImageText']) > 0:
                    table_image_text = t_json['ImageText']
                    # sort cells by up
                    sorted_table = sorted(table_image_text,
                                          key=lambda cell: (cell['TextBB'][1] + cell['TextBB'][3]) / 2.0)

                    # calculate total "unique" ups
                    ups = [x['TextBB'][1] for x in sorted_table]
                    ups = sorted(list(set(ups)))  # unique ups
                    up2rootup = {ups[0]: ups[0]}
                    for i in range(0, len(ups) - 1):
                        if ups[i + 1] - up2rootup[ups[i]] > 0.5:
                            up2rootup[ups[i + 1]] = ups[i + 1]
                        else:
                            up2rootup[ups[i + 1]] = ups[i]

                    # construct a dict with up as key
                    row_dict = defaultdict(list)
                    for cell in sorted_table:
                        up = up2rootup[cell['TextBB'][1]]
                        row_dict[up].append(cell)

                    # merge mergeable cells in each row
                    merged_cells = []
                    cnt = 0
                    for row in row_dict.values():
                        sorted_row = sorted(row, key=lambda cell: cell['TextBB'][2])  # sorted by right

                        root_dict = {0: 0}
                        for i in range(len(sorted_row) - 1):
                            if (sorted_row[i + 1]['TextBB'][0] - sorted_row[i]['TextBB'][2]) < 5:  # width for space
                                root_dict[i + 1] = root_dict[i]
                            else:
                                root_dict[i + 1] = i + 1

                        inverse_root_dict = defaultdict(list)
                        for ind, root in root_dict.items():
                            inverse_root_dict[root].append(ind)

                        for root, inds in inverse_root_dict.items():
                            ind = inds[0]
                            cell = sorted_row[ind]
                            for i in range(1, len(inds)):
                                cell = merge_cell(cell, sorted_row[inds[i]])
                            merged_cells.append(cell)
                    calib_table_json[k]['ImageText'] = merged_cells

            for k, t_json in enumerate(calib_table_json):
                calib_t_json = deepcopy(t_json)
                cells = calib_t_json['ImageText']
                for cell in cells:
                    cell['TextBBCenter'] = [round((cell['TextBB'][0] + cell['TextBB'][2]) / 2, 3),
                                            round((cell['TextBB'][1] + cell['TextBB'][3]) / 2, 3)]
                    if re.search(pattern, cell['Text']):
                        cell['Pattern'] = 1
                    else:
                        cell['Pattern'] = 0

                center_horizontal_bin = defaultdict(list)
                center_vertical_bin = defaultdict(list)
                left_vertical_bin = defaultdict(list)
                right_vertical_bin = defaultdict(list)

                for cell in cells:
                    if cell['Pattern']:
                        # cell_text = re.sub(r"([.,!:?()])", r" \1 ", cell['Text'])
                        # cell_texts = []
                        # for t in cell_text.split():
                        #     if re.match(exact_pattern, t):
                        #         cell_texts.append(t)
                        # if len(cell_texts) != 1:
                        #     continue
                        # cell['Text'] = cell_texts[0]

                        center_horizontal_bin[cell['TextBBCenter'][0]].append(cell)
                        center_vertical_bin[cell['TextBBCenter'][1]].append(cell)
                        left_vertical_bin[cell['TextBB'][0]].append(cell)
                        right_vertical_bin[cell['TextBB'][2]].append(cell)
                        valid_cell_texts.append(cell['Text'])

                calib_t_json['VerticalBin'] = center_vertical_bin
                calib_t_json['HorizontalBin'] = center_horizontal_bin
                calib_t_json['LeftVerticalBin'] = left_vertical_bin
                calib_t_json['RightVerticalBin'] = right_vertical_bin
                calib_t_json['ImageText'] = ''
                calib_table_json[k] = calib_t_json

            tables[file_prefix] = calib_table_json

    # tables = dict(tables)
    with (dataset_dir / 'tables.json').open(mode='w') as fout:
        fout.write(json.dumps(dict(tables)))

    # elements in the same row / column

    # a black list to deal with common capitalized non-tech words
    black_list = set(['METHOD', 'LOCATION', 'TIME', 'DATASET', 'DATA', 'SET', 'METRIC', 'NA', \
                      'YES', 'NO',
                      'ETHOD', 'OCATION', 'IME', 'ETHODS'])

    with (dataset_dir / 'table_align_clean.txt').open(mode='w') as fout:
        for filename, table_json in tables.items():
            for t_json in table_json:
                vertical_bin = t_json['VerticalBin']
                horizontal_bin = t_json['HorizontalBin']
                left_vertical_bin = t_json['LeftVerticalBin']
                right_vertical_bin = t_json['RightVerticalBin']

                for bin_ in [vertical_bin, horizontal_bin, left_vertical_bin, right_vertical_bin]:
                    for cells in bin_.values():
                        if len(cells) >= 2:
                            acros = []
                            for cell in cells:
                                acro = cell['Text']
                                acro = re.sub('\(.*\)', '', acro).strip()
                                acro = re.sub('\[.*\]', '', acro).strip()
                                acro = re.sub('-?\d+$', '', acro).strip()
                                acro = re.sub('^-', '', acro)
                                acro = acro.strip(',').strip('*')
                                if re.search('[/+ ]', acro):
                                    continue
                                if acro in black_list:
                                    continue
                                if not re.search(exact_pattern, acro):
                                    continue
                                acros.append(acro)
                            acros = list(set(acros))
                            if len(acros) >= 2:
                                # # remove capital words
                                # flag = True
                                # for a in acros:
                                #     if not re.search('^[A-Z]+$', a):
                                #         flag = False
                                #         break
                                #
                                # if flag:
                                #     for a in acros:
                                #         if len(wn.synsets(a)) == 0:
                                #             flag = False
                                #             break
                                # if flag:
                                #     continue
                                # if 'NORTH' in acros:
                                #     import pdb; pdb.set_trace();
                                # fout.write(filename+'\t')
                                fout.write(' '.join(acros))
                                fout.write('\n')

    # # get co-occurrence statistics from table
    # table_abbv_hist = Counter()
    # co_occur = defaultdict(int)
    # abbvfile = '../data/{}_table_align_clean.txt'.format(dataset)
    # with open(abbvfile) as fin:
    #     for line in fin:
    #         line = line.strip()
    #         if line:
    #             abbvs = line.split(' ')
    #             for w in abbvs:
    #                 table_abbv_hist[w] += 1
    #             pairs = all_pairs(abbvs)
    #             for w1, w2 in pairs:
    #                 co_occur[(w1, w2)] += 1
    #                 co_occur[(w2, w1)] += 1

    session = DBSession()
    suffix = '.pdf_clean.txt'
    suffix_len = len(suffix)
    docname2id = {}
    id2docname = {}
    for doc in session.query(Document).all():
        docname = doc.filename[:-suffix_len]
        docname2id[docname] = doc.id
        id2docname[doc.id] = docname
    session.close()

    docid2tableacro = defaultdict(list)
    for docname, doc_json in tables.items():
        docid = docname2id[docname]
        for t_json in doc_json:
            vertical_bin = t_json['VerticalBin']
            horizontal_bin = t_json['HorizontalBin']
            for cells in vertical_bin.values():
                if len(cells) >= 2:
                    acros = []
                    for cell in cells:
                        acro = cell['Text']
                        acro = re.sub('\(.*\)', '', acro)
                        acro = acro.strip()
                        if re.search('[/+ ]', acro):
                            continue
                    docid2tableacro[docid].append(acros)

    with (dataset_dir / 'docid2tableacro.json').open(mode='w') as fout:
        fout.write(json.dumps(dict(docid2tableacro)))
