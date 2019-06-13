# coding=utf-8
"""Convert pdf files to txt files"""

import argparse
import os
import subprocess
from multiprocessing import Pool
from tqdm import tqdm


def convert2txt(args):
    filepath, savefilepath = args
    if os.path.exists(savefilepath) is False:
        subprocess.call(['pdftotext', filepath, savefilepath])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset name")
    args = parser.parse_args()

    #  NIPS/ACL/VLDB dataset
    dataset = args.dataset
    pdfrootpath = '../data/PaperCrawler/{dataset}/{dataset}paper'.format(
        dataset=dataset)
    txtrootpath = '../data/PaperCrawler/{dataset}/{dataset}papertxt'.format(
        dataset=dataset)

    if os.path.exists(txtrootpath) is False:
        os.mkdir(txtrootpath)

    dirlist = os.listdir(pdfrootpath)

    for yeardir in tqdm(dirlist):  # dir structure: NIPSpaper/NIPS1999/...
        filelist = os.listdir(pdfrootpath + '/' + yeardir)
        filedir = pdfrootpath + '/' + yeardir
        savedir = txtrootpath + '/' + yeardir
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        argslist = []
        for file in tqdm(filelist):
            filepath = filedir + '/' + file
            savefilepath = savedir + '/' + file + '.txt'
            args = filepath, savefilepath
            argslist.append(args)

        p = Pool(30)
        p.map(convert2txt, argslist)
        p.close()
        p.join()
