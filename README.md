# AlgMap
The source code used for the KDD'2019 paper Mining Algorithm Roadmap in Scientific Publications.

### Requirements
```
Python 3.6
pytorch 1.0
sqlite3
pdffigures, https://github.com/allenai/pdffigures
Other python requirements, pip install -r requirements.txt
```

### Preprocessing
Codes in preprocess folder prepare data needed for constructing Algorithm Roadmap.
In the experiment, datasets are NIPS, ACL and VLDB.

```
cd preprocess/
```

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
python build_vocab.py ${dataset}
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
and data splitting for final relation extraction. Prepare instances for acronym pairs, feature_type standard or paragraph 
denote single-sentence or cross-sentence instances.
```
python parse_abbv_in_sents.py ${dataset}
python prepare_final_data.py ${dataset} ${feature_type}
```

### Pre-processed Dataset
Pre-processed NIPS, ACL, VLDB datasets before running prepare_final_data.py.
```
https://drive.google.com/open?id=1R00G5xP141SO5oCt9zL8-COsX5G2H8dy
```


### Train And Test

#### Train Model
Train our CANTOR relation extraction model.
```
python train_cantor.py ${dataset} cantor ${feature_type}
```
    
#### Evaluate Model
Evaluate model and save predictions.
```
python model_eval.py ${dataset} cantor ${feature_type}
```

#### Construct Algorithm Roadmap
Construct algorithm roadmap for a query.
```
python construct_roadmap.py 
```