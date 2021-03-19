# RENET2 -  High-Performance Full-Text Gene-Disease Relation Extraction with Iterative Training Data Expansion

---
![RENET2 logo](/data/image/RENET2_logo.png "RENET2 logo")

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Contact: Junhao Su \
Email: jhsu@cs.hku.hk


## Introduction

Relation extraction is a fundamental task for extracting gene-disease associations from biomedical text. Existing tools have limited capacity, as they can only extract gene-disease associations from single sentences or abstract texts. In this work, we propose RENET2, a deep learning-based relation extraction method, which implements section filtering and ambiguous relations modeling to extract gene-disease associations from full-text articles. RENET2 adopted a novel iterative data expansion strategy to build an annotated full-text dataset to resolve the scarcity of labels on full-text articles.

---

## Contents

- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Download Data and Trained Models](#download-data-and-trained-models)
- [Run RENET2 Model](#run-renet2-model)
- [Understand Output File](#understand-output-file)
- [Dataset](#dataset)
- [Modules Descriptions](#modules-descriptions)
- [Benchmark](#benchmark)
- [(Optional) Data Preparing](#optional-data-preparing)
- [(Optional) Visualization](#optional-visualization)
- [(Manually) Download Data and Trained Models](#manually-download-data-and-trained-models)

---

## Folder Structure

    .
    ├── renet2              # RENET2 scource codes
    ├── data                # RENET2 data
    ├── models              # RENET2 trained models
    ├── tools               # Utils tools
    ├── benchmark           # Benchmark codes for BeFree/DTMiner/BioBERT
    ├── test                # Addtional test dir
    └── README.md           

## Installation

### Option 1: Install RENET2 from GitHub 

```bash
git clone https://github.com/sujunhao/RENET2.git
cd RENET2 

conda create -n renet2 python=3.7
conda activate renet2

conda install -c conda-forge ruby scikit-learn=0.22.2.post1 pandas=1.0.1 numpy=1.18.1 tqdm=4.42.1
conda install pytorch==1.2.0 cudatoolkit=10.0 -c pytorch

#install genia sentence splitter from: http://www.nactem.ac.uk/y-matsu/geniass/
cd tools
# wget http://www.nactem.ac.uk/y-matsu/geniass/geniass-1.00.tar.gz
tar -xf geniass-1.00.tar.gz
cd geniass
make
cd ../../
```

### Option 2: Install RENET2 from Bioconda

(TBD)


## Download Data and Trained Models

Download all required files

```
# download data and models
# make sure you are in the root dir of RENET2
bash renet2/download_renet2_data.sh
```

Or download individual files via [(manually) Download Data and Trained Models](#manually-download-data-and-trained-models)


Quick test after downloaded all required files

```
# quick testing
python renet2/predict_renet2_ft.py --raw_data_dir data/ft_data/ --gda_fn_d data/ft_gda/ --models_number 4 --batch_size 8 --max_doc_num 10 --no_cache_file --use_fix_pretrained_models


# check predicted results
# predicted gene-disease associations saved in data/ft_gda/gda_rst.tsv
less data/ft_gda/gda_rst.tsv
```

## Run RENET2 Model

```
## make sure you downloaded data and models at the [Download Data and Trained Models] section
## use --help to check more information
## use --use_cuda if you want to use GPUs


# train 10 RENET2 models (optional, trained model already in the models dir)
python renet2/train_renet2_ft.py --raw_data_dir data/ft_data/ --annotation_info_dir data/ft_info --model_dir models/ft_models/ --pretrained_model_p models/Bst_abs_10  --epochs 10 --models_number 10 --batch_size 60 --have_SiDa data/ft_info/ft_base/ft_base --gda_fn_d data/ft_gda/ --use_cuda


# use trained RENET2 models to predict GDAs (using --is_sensitive_mode to enable RENET2-Sensitive mode)
python renet2/predict_renet2_ft.py --raw_data_dir data/ft_data/ --model_dir models/ft_models/ --models_number 10 --batch_size 60 --gda_fn_d data/ft_gda/ --use_cuda

# check predicted GDAs
less ../data/ft_gda/gda_rst.tsv

# apply 5-fold cross-validation to test RENET2 performance
python renet2/evaluate_renet2_ft_cv.py --epochs 10 --raw_data_dir data/ft_data/ --annotation_info_dir data/ft_info/ --rst_file_prefix ft_base --have_SiDa data/ft_info/ft_base/ft_base --pretrained_model_p models/Bst_abs_10 --no_cache_file --use_cuda
```

### Pipeline: Use RENET2 to Predict Gene-Disease Associations from Articles ID

    Input: PMID and PMCID list          [example: RENET2/data/test/test_download_pmcid_list.csv]
    Output: Gene-Disease Assoications   [example: will generate at RENET2/data/test_data/gda_rst.tsv]
    
pipeline with example

Input data: PMID and PMCID list `RENET2/data/test/test_download_pmcid_list.csv`

1. download text and NER annotations  
```
# download abstract and its annotations
# (download abstract is required for the full-text case, as some full-text at PTC did not have an abstract section, should download separately)
python renet2/download_data.py --process_n 3 --id_f test/test_download_pmcid_list.csv --type abs --dir data/raw_data/abs/ --tmp_hit_f data/test_data/hit_id_l.csv

# download full-text and its annotations
python renet2/download_data.py --process_n 3 --id_f test/test_download_pmcid_list.csv --type ft --dir data/raw_data/ft/ --tmp_hit_f data/test_data/hit_id_l.csv
```
2. parse text and enetities annotations to RENET2 input format
```
# parse data
python renet2/parse_data.py --id_f test/test_download_pmcid_list.csv --type 'ft' --in_abs_dir data/raw_data/abs/  --in_ft_dir data/raw_data/ft/ --out_dir data/test_data/

# normalize NET ID (optinal) 
python renet2/normalize_ann.py --in_f data/test_data/anns.txt --out_f data/test_data/anns_n.txt
```
3. run RENET2 on parsed data
```
python renet2/predict_renet2_ft.py --raw_data_dir data/test_data/ --model_dir models/ft_models/ --gda_fn_d data/test_data/ --models_number 4 --batch_size 8 --max_doc_num 10 --no_cache_file 
```
Output data: predicted Gene-Disease Associations are stored in `RENET2/data/test_data/gda_rst.tsv`



## Understand Output File

There are 7 columns in the gda_rst.tsv:  

| 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | 
| pmid    | geneId  | diseaseId   |    g_name  | d_name  | prob_avg     |   prob_X |

where `pmid` is Article PubMed Id, `geneId` is the Entrez Gene ID (Entrez), `diseaseId` is the Disease Id (MESH), `g_name` is the gene name (a ID with multiple names will be seperated by '|'), `d_name` is the disease name (a ID with multiple names is seperated by '|'), `prob_avg` is the predicted mean GDP (gene-disease probability) of all 10 models, 
`prob_X` is the predicted GDP of each models.

## Dataset

### Parsed Dataset for RENET2

Make sure you downloaded data at the [Download Data and Trained Models] section.

    .
    ├── data                    
    │   ├── ft_data             # full-text dataset
    │   │   ├── docs.txt        # articles with ID/title/abstract/main text
    │   │   ├── sentences.txt   # sentences from articles [collected from geniass]
    │   │   ├── anns.txt        # gene/disease annotations [collected from PubTator Central]
    │   │   ├── anns_n.txt      # gene/disease annotations with normalize annotated ID 
    │   │   ├── labels.txt      # gene-disease assoications table 
    │   │   └── s_docs.txt      # articles with section's ID (for visualization of annotated results)
    │   ├── abs_data            # abstract dataset
    │   │   ├── 1st_ann         # abstract dataset, first round
    │   │   │   └── ...                 
    │   │   ├── 2nd_ann         # abstract dataset, second round [Abstract-exp in paper]
    │   │   │   └── ...                 
    │   │   ├── ori             # training dataset from RENET
    │   │   │   └── ...                 
    │   │   └── ori_test        # testing dataset from RENET
    │   │       └── ...                 
    │   └── ...                 
    └── ...

### Annotated Gene-disease Associations Based on Iterative Training Data Expansion

Annotated gene-disease associations based on iterative training data expansion strategy. These are the original annotation files, the parsed files are located at the parsed dataset, please check it accordingly.

    .
    ├── data                    
    │   ├── ft_info             
    │   │   └── ft_500_n.tsv    # annotated full-text GDA (fisrt and second round)
    │   ├── ann_table          
    │   │   ├── ann_1st.tsv     # annotated abstract GDA (fisrt round)
    │   │   └── ann_2nd.tsv     # annotated abstract GDA (second round)
    │   └── ...                 
    └── ...


### Found Gene-disease Associations from PMC & LitCovid

Make sure you downloaded data at the [Download Data and Trained Models] section.

    .
    ├── data                    
    │   ├── pmc 
    │   │   └──gda_rst.tsv    # GDA from PMC
    │   ├── litcovid 
    │   │   └──gda_rst.tsv    # GDA from LitCovid
    │   └── ...                 
    └── ...

## Modules Descriptions

Modules in `renet2/` are for model training/testing.

*For the Modules listed below, please use the `-h` or `--help` option for checking available options.*

`renet2/` | source code for RENET2 
---: | ---
`train_renet2_ft.py` | Module for training RENET2 models.
`predict_renet2_ft.py` | Using RENET2 models to predict gene-disease associations.
`evaluate_renet2_ft_cv.py` | Evaluating trained RENET2 models and using cross-validation.
`train_renet2_ft_cv.py` | Training RENET2 models and using cross-validation to evaluating models performance.
`download_data.py` | Downloading articles from PMC/PTC with provided PMID/PMCID list. (please check example an RENET2/renet2/pre_precoss/)
`parse_data.py` | Parsing articles from RENET2. (please check example an RENET2/renet2/pre_precoss/)


## Benchmark

### Run BeFree

```
pip install pymongo
pip install regex
cd benchmark/BeFree
git clone git@bitbucket.org:ibi_group/befree.git

wget http://www.bio8.cs.hku.hk/RENET2/renet2_bm_befree.tar.gz
tar -xf renet2_bm_befree.tar.gz
#run Generate_BeFree_Input.ipynb on python jypyter notebook to genrate BeFree input
sh benchmark_befree.sh
```

### Run DTMiner
```
cd benchmark/DTMiner
wget http://www.bio8.cs.hku.hk/RENET2/renet2_bm_dtminer.tar.gz
tar -xf renet2_bm_dtminer.tar.gz
#run Generate_DTMiner_Input.ipynb on python jypyter notebook to genrate BeFree input
sh benchmark_DTMiner.sh 
```

### Run BioBERT
```
cd benchmark/BioBERT
git clone https://github.com/dmis-lab/biobert
cd biobert; pip install -r requirements.txt
./download.sh

# generate BioBERT input
run Generate_BioBERT_Input.ipynb on python jypyter notebook

# run BioBERT
sh run_bert.sh
```

### Benchmark (Using RENET2 Cross-validation to Evalutate RENET2/BeFree/DTMiner/BioBERT Results)

```

python evaluate_renet2_ft_cv.py --epochs 10 --raw_data_dir ../data/ft_data/  --rst_file_prefix ft_base --have_SiDa ../data/ft_info/ft_base/ft_base
```

### Benchmark RENET2 on Abstract Data

```
# training RENET2 model on abstract data
run ./renet2/build_best_model_abs.ipynb on jupyter notebook
# Using cross-validation to benchmarking RENET2 on abstract data
run ./renet2/exp_abs.ipynb on jupyter notebook
```


## (Optional) Data Preparing

Download/parse data from PubMed/PubTator Central

Please provide your list at `RENET2/data dir`, examples of pmid list can be found at `RENET2/data/test_download_pmid_list.csv` (for abstract) or `RENET2/data/test_download_pmcid_list.csv` (for full-text).

```
cd renet2

# downloading data
# testing download abstracts data 
python renet2/download_data.py --process_n 3 --id_f test/test_download_pmcid_list.csv --type abs --dir data/raw_data/abs/ --tmp_hit_f data/test_data/hit_id_l.csv
# testing download full-text data
python renet2/download_data.py --process_n 3 --id_f test/test_download_pmcid_list.csv --type ft --dir data/raw_data/ft/ --tmp_hit_f data/test_data/hit_id_l.csv

# parsing data
python renet2/parse_data.py --id_f test/test_download_pmcid_list.csv --type 'ft' --in_abs_dir data/raw_data/abs/  --in_ft_dir data/raw_data/ft/ --out_dir data/test_data/
# (optional) normalize annotated ID
python renet2/normalize_ann.py --in_f data/test_data/anns.txt --out_f data/test_data/anns_n.txt
```          

## (Optional) Visualization

Found and visualze a pair of gene-disease annotation obtrained from Pubtar Central.


```
run ./renet2/vis_text.ipynb on jupyter notebook
```
![RENET2 vis](/data/image/RENET2_vis.png "RENET2 vis")


## (Manually) Download Data and Trained Models

```
# download data
## download abstract dataset [dir: renet2/data/abs_data]
cd data/abs_data
wget http://www.bio8.cs.hku.hk/RENET2/renet2_abs_data.tar.gz
tar -xf renet2_abs_data.tar.gz
cd ../..

## download full-text dataset [dir: renet2/data/ft_data]
cd data/ft_data
wget http://www.bio8.cs.hku.hk/RENET2/renet2_ft_data.tar.gz
tar -xf renet2_ft_data.tar.gz
cd ../..

## download gene-disease assoications found from PMC and LitCovid [dir: renet2/data/pmc_litcovid]
cd data
wget http://www.bio8.cs.hku.hk/RENET2/renet2_gda_pmc_litcovid_rst.tar.gz
tar -xf renet2_gda_pmc_litcovid_rst.tar.gz
cd ..


# download trained model
## download trained abstract model [dir: renet2/models/abs_models]
cd models
wget http://www.bio8.cs.hku.hk/RENET2/renet2_trained_models_abs.tar.gz
tar -xf renet2_trained_models_abs.tar.gz
cd ..

## download trained abstract model [dir: renet2/models/ft_models]
cd models
wget http://www.bio8.cs.hku.hk/RENET2/renet2_trained_models_ft.tar.gz
tar -xf renet2_trained_models_ft.tar.gz
cd ../..

```
