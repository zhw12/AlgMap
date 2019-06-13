# coding: utf-8
"""
    Parse pdf files with pdffigures tool
"""

import argparse
import subprocess
from pathlib import Path

from tqdm import tqdm

# tablejson is finally used for table supervision
parser = argparse.ArgumentParser()
parser.add_argument("option", help="fig | colorfig | tablejson")
parser.add_argument("dataset", help="dataset name")
args = parser.parse_args()
option = args.option
dataset = Path(args.dataset)  # e.g. ACL

rootdir = Path('/home/hanwen/techmap/data/PaperCrawler')
pdf_rootdir = rootdir / dataset / (dataset.name + 'paper')

if option == 'fig':
    output_rootdir = rootdir / dataset / (dataset.name + 'paper' + 'fig')
elif option == 'colorfig':
    output_rootdir = rootdir / dataset / (dataset.name + 'paper' + 'colorfig')
elif option == 'tablejson':
    output_rootdir = rootdir / dataset / (dataset.name + 'paper' + 'tablejson')

import pdb;

pdb.set_trace();
output_rootdir.mkdir(parents=True, exist_ok=True)

dataset_yearlist = sorted(pdf_rootdir.iterdir(), reverse=False)

for year in tqdm(dataset_yearlist, desc='year'):
    p = output_rootdir / year.name
    p.mkdir(parents=True, exist_ok=True)
    for filename in tqdm(sorted(list((pdf_rootdir / year).iterdir())), leave=False, desc='file'):
        pdf_file = pdf_rootdir / year.name / filename
        output_fileprefix = output_rootdir / year.name / pdf_file.name[:-4]

        if option == 'fig':
            subprocess.call(['pdffigures', '-o', output_fileprefix, '-i', pdf_file])
        elif option == 'colorfig':
            subprocess.call(['pdffigures', '-c', output_fileprefix, '-i', pdf_file])
        elif option == 'tablejson':
            subprocess.call(['pdffigures', '--save-json', output_fileprefix, '-i', pdf_file])
