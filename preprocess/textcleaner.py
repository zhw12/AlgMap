# coding=utf-8
"""Remove special non-ascii tokens from converted txt files"""

import argparse
import re
import string
from pathlib import Path

from tqdm import tqdm


def clean_text(text):
    # data = re.sub('\(cid:\d+\)\s*', '', text)
    cleaned_text = re.sub(r'-\s+', '', text)
    cleaned_text = re.sub('\n(?=[^\n])', ' ', cleaned_text)
    # data = re.sub('- ', '', data) # for cases like Soci- ety, ne-ural
    # data = filter(lambda x: x in string.printable, data)
    printable_set = set(string.printable)
    cleaned_text = ''.join([x for x in cleaned_text if x in printable_set])
    cleaned_text = re.sub(r'\[[\d, ]+\]', '<CIT>', cleaned_text)
    cleaned_data = []
    for line in cleaned_text.split('\n'):
        line = line.strip()
        if line:
            total_chars = len(line)
            total_words = len(re.split('[^a-zA-Z]', line))
            if float(total_chars + 1) / float(total_words) < 3:
                pass
            else:
                cleaned_data.append(line)
    cleaned_text = '\n'.join(cleaned_data)
    cleaned_text = re.sub(r'\b[\d.e]+\b', '<NUM>', cleaned_text)
    return cleaned_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset name")
    args = parser.parse_args()

    #  NIPS/ACL/VLDB dataset
    dataset = args.dataset
    txt_rootdir = Path('../data/PaperCrawler') / dataset / (dataset + 'papertxt')
    cleantxt_rootdir = Path('../data/PaperCrawler') / dataset / (dataset + 'papertxtClean')

    dataset_yearlist = sorted(txt_rootdir.iterdir(), reverse=True)

    for yeardir in tqdm(dataset_yearlist):
        savedir = cleantxt_rootdir / yeardir.name
        savedir.mkdir(parents=True, exist_ok=True)
        for file in tqdm(yeardir.iterdir()):
            savefile = savedir / (file.name[:-4] + '_clean.txt')
            with file.open() as fin:
                data = fin.read()
            cleanedText = clean_text(data)
            with savefile.open(mode='w') as fout:
                fout.write(cleanedText)
