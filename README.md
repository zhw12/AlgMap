# AlgMap
The source code used for the paper Mining Algorithm Roadmap in Scientific Publications.

### Environments
Ubuntu 16.04 LTS,
Python 3,
GPU environment.
sqlite3

### Requirements
```
pip install -r requirements.txt
pdffigures, https://github.com/allenai/pdffigures
```

### Preprocessing
Codes in **preprocess folder** prepare data needed for constructing Algorithm Roadmap.

```
cd preprocess/
```


<!--
#### Crawl Corpus
To avoid copyright issue, provide example crawling script from original website.
```
python crawler.py
```
-->

#### Convert Pdf To Text
Set the correct data path and use the following script to batchly execute pdftotext.
```
python convert2txt_multithread.py ${dataset}
```
    

#### Clean Text Corpus
Clean the converted txt files, remove non-ascii characters etc. 
```
python textcleaner.py ${dataset}
```

#### Extract Table Json Files From Pdf
Extract tables from pdf files for later weak supervision extraction, relying on pdffigures.
```
python pdf2figure.py tablejson ${dataset}
```
    
#### Create Database
Use sqlite3 database to store the corpus for later instance collection.
```
python create_db.py ${dataset}
python concat_docs_in_one_file.py ${dataset}
```

#### Pre-train Word Embedding
```
python pretrain_wordvec.py ${dataset}
```
	
	
#### Build Vocabulary
```
python build_vocab.py NIPS (1min)
```
    


#### Parse Table Json Files
Parse table json files to extract weak supervision.
```
python parse_table.py ${dataset}
```
	
#### Save Co-occurred Acronyms
Save co-occured acronyms which are baseline methods used in the experiment.
```
python save_coocurred_abbvs.py ${dataset}
```
    

#### Prepare Final Data
Cut off long paragraphs and long sentences, do labelling, padding, 
and data splitting for final relation extraction.
```
python parse_abbv_in_sents.py ${dataset}
python prepare_final_data.py ${dataset} standard
python prepare_final_data.py ${dataset} paragraph 
```


#### Train Model
Training our CANTOR relation extraction model.
```
python train.py main 1 ${dataset} ${model_name}
```
    
