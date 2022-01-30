# RENET2: High-Performance Full-text Gene-Disease Relation Extraction with Iterative Training Data Expansion

---
![RENET2 logo](/repo_f/RENET2_logo.png "RENET2 logo")

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) [![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat)](http://bioconda.github.io/recipes/renet2/README.html)

Contact: Junhao Su \
Email: jhsu@cs.hku.hk


## Introduction

Relation extraction (RE) is a fundamental task for extracting gene–disease associations from biomedical text. Many state-of-the-art tools have limited capacity, as they can extract gene–disease associations only from single sentences or abstract texts. A few studies have explored extracting gene–disease associations from full-text articles, but there exists a large room for improvements. In this work, we propose RENET2, a deep learning-based RE method, which implements Section Filtering and ambiguous relations modeling to extract gene–disease associations from full-text articles. We designed a novel iterative training data expansion strategy to build an annotated full-text dataset to resolve the scarcity of labels on full-text articles. In our experiments, RENET2 achieved an F1-score of 72.13% for extracting gene–disease associations from an annotated full-text dataset, which was 27.22, 30.30, 29.24 and 23.87% higher than BeFree, DTMiner, BioBERT and RENET, respectively. We applied RENET2 to (i) ∼1.89M full-text articles from PubMed Central and found ∼3.72M gene–disease associations; and (ii) the LitCovid articles and ranked the top 15 proteins associated with COVID-19, supported by recent articles. RENET2 is an efficient and accurate method for full-text gene–disease association extraction. The source-code, manually curated abstract/full-text training data, and results of RENET2 are available at this repo.


RENET2 is published in [*NAR Genomics and Bioinformatics*](https://academic.oup.com/nargab/article/3/3/lqab062/6315218).


---

## Contents

- [What's new](#whats-new)
- [Reference for application](#reference-for-application)
- [Installation](#installation)
- [Download Data and Trained Models](#download-data-and-trained-models)
- [Usage](#usage)
- [Understand Output File](#understand-output-file)
- [Dataset](#dataset)
- [Modules Descriptions](#modules-descriptions)
- [Benchmark](#benchmark)
- [(Optional) Visualization](#optional-visualization)

---

## What's new?
- 20210716
    
    The paper of RENET2 is published. We updated and fixed the empty parsed dataset problem, and updated the parsed full-text dataset in data/ft_data.
    
- 20210514

    Update README with data link: [http://www.bio8.cs.hku.hk/RENET2/renet2_data_models.tar.gz](http://www.bio8.cs.hku.hk/RENET2/renet2_data_models.tar.gz). The full-test annotated dataset is available at `/data/ft_info folder` in the download files. Please check this [link1](#download-data-and-trained-models) and [link2](#annotated-gene-disease-associations-based-on-iterative-training-data-expansion) for more detail.
    
    Add RENET testing script for full-text dataset

- 20210330

    We can install RENET2 via bioconda now! and the code for the RENET2 is refined as a python package.
    
## Reference for Application:

- Microsoft's [BiomedNLP-PubMedBERT](https://huggingface.co/jambo/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-finetuned-renet), from [James Morrill](https://github.com/jambo6). It achieves an F1 score of 0.8 at the abstract dataset.


## Installation

### Option 1: Install RENET2 from Bioconda 
```bash

conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge
# create conda environment named "renet2-env"
conda create -n renet2-env -c bioconda renet2
conda activate renet2-env

# run renet2 like this afterwards
renet2 --help
```

### Option 2: Install RENET2 from GitHub

```bash
# create renet2 env
conda create -n renet2-env python=3.7
conda activate renet2-env

# install required package
conda install -c conda-forge ruby scikit-learn=0.22.2.post1 pandas=1.0.1 numpy=1.18.1 tqdm=4.42.1
conda install pytorch==1.2.0 cudatoolkit=10.0 -c pytorch

git clone https://github.com/sujunhao/RENET2.git
cd RENET2 
pip install . --no-deps --ignore-installed

# run renet2 like this afterwards
renet2 --help
```

## Download Data and Trained Models

Download all required files

All data and models are available at this link: [http://www.bio8.cs.hku.hk/RENET2/renet2_data_models.tar.gz](http://www.bio8.cs.hku.hk/RENET2/renet2_data_models.tar.gz), please using the following scripts to download data for RENET2.

```
### if RENET2 is installed from Bioconda
mkdir RENET2
cd RENET2
RENET2_DATA_S_URL=https://raw.githubusercontent.com/sujunhao/RENET2/main/src/renet2/download_renet2_data.sh
curl -s ${RENET2_DATA_S_URL} | bash -s .
R2_DIR=$(pwd)

### if RENET2 is installed from GitHub
#### make sure you are in the root dir of RENET2
bash src/renet2/download_renet2_data.sh .
R2_DIR=$(pwd)
```

### Quick test after downloaded all required files

```
# quick testing
# R2_DIR="[DATA/MODEL_PATH]"                                     # e.g. ~/git/RENET2, check 'Download Data and Trained Models'
renet2 predict --raw_data_dir ${R2_DIR}/data/ft_data/ --gda_fn_d ${R2_DIR}/data/ft_gda/ --models_number 4 --batch_size 8 --max_doc_num 10 --no_cache_file  --model_dir ${R2_DIR}/models/ft_models/

# check predicted results
# predicted gene-disease associations
less ${R2_DIR}/data/ft_gda/gda_rst.tsv
```

## Usage

### General usage

```
# help page for renet2
renet2 --help

# to run a submodule using python
renet2 [submodule] [options]
```

### Setup variables for renet2

```
R2_DIR="[DATA_MODEL_PATH]"                                     # e.g. ~/git/RENET2, check 'Download Data and Trained Models'
```

### Run RENET2 Model

```
## for using RENET2, please make sure that -
## 1. in RENET2-env environment (using 'conda activate RENET2-env' to setup RENET2 environment)
## 2. follow the 'Download Data and Trained Models' to download RENET2 dataset and trained models first
## 3. setup the `R2_DIR` variable as in 'Setup variables for renet2'
## use --use_cuda if you have GPUs and want to use GPUs

# set RENET2's models dir, noted that trained model already in this dir
MODEL_DIR=${R2_DIR}/models/ft_models

# train 10 RENET2 models (optional, trained model already in the models dir)
MODEL_DIR=${R2_DIR}/models/ft_models_test
renet2 train --raw_data_dir ${R2_DIR}/data/ft_data/ --annotation_info_dir ${R2_DIR}/data/ft_info --model_dir ${MODEL_DIR} --pretrained_model_p ${R2_DIR}/models/Bst_abs_10  --epochs 10 --models_number 10 --batch_size 60 --have_SiDa ${R2_DIR}/data/ft_info/ft_base/ft_base --gda_fn_d ${R2_DIR}/data/ft_gda/ --use_cuda

# use trained RENET2 models to predict GDAs (using --is_sensitive_mode to enable RENET2-Sensitive mode)
# maximum using 10 models to predict
renet2 predict --raw_data_dir ${R2_DIR}/data/ft_data/ --model_dir ${MODEL_DIR} --models_number 2 --batch_size 60 --gda_fn_d ${R2_DIR}/data/ft_gda/ --use_cuda

# check predicted GDAs
less ${R2_DIR}/data/ft_gda/gda_rst.tsv

# apply 5-fold cross-validation to test RENET2 performance
renet2 evaluate_renet2_ft_cv --epochs 10 --raw_data_dir ${R2_DIR}/data/ft_data/ --annotation_info_dir ${R2_DIR}/data/ft_info/ --rst_file_prefix ft_base --have_SiDa ${R2_DIR}/data/ft_info/ft_base/ft_base --pretrained_model_p ${R2_DIR}/models/Bst_abs_10 --no_cache_file --use_cuda
```

### Pipeline: Use RENET2 to predict Gene-Disease Associations from articles ID

    Input: PMID and PMCID list          [example: RENET2/test/test_download_pmcid_list.csv]
    Output: Gene-Disease Assoications   [example: will generate at RENET2/data/test_data/gda_rst.tsv]
    
pipeline with example 

Input data: PMID and PMCID list `${R2_DIR}/test/test_download_pmcid_list.csv`

1. download text and NER annotations  
```
# download abstract and its annotations
# (download abstract is required for the full-text case, as some full-text at PTC did not have an abstract section, should download separately)
renet2 download_data --process_n 3 --id_f ${R2_DIR}/test/test_download_pmcid_list.csv --type abs --dir ${R2_DIR}/data/raw_data/abs/ --tmp_hit_f ${R2_DIR}/data/test_data/hit_id_l.csv

# download full-text and its annotations
renet2 download_data --process_n 3 --id_f ${R2_DIR}/test/test_download_pmcid_list.csv --type ft --dir ${R2_DIR}/data/raw_data/ft/ --tmp_hit_f ${R2_DIR}/data/test_data/hit_id_l.csv
```

2. parse text and enetities annotations to RENET2 input format
```
# parse data
renet2 install_geniass          # install geniass, only run one time
conda install ruby              # install ruby
renet2 parse_data --id_f ${R2_DIR}/test/test_download_pmcid_list.csv --type 'ft' --in_abs_dir ${R2_DIR}/data/raw_data/abs/  --in_ft_dir ${R2_DIR}/data/raw_data/ft/ --out_dir ${R2_DIR}/data/test_data/

# normalize NET ID
renet2 normalize_ann  --in_f ${R2_DIR}/data/test_data/anns.txt  --out_f ${R2_DIR}/data/test_data/anns_n.txt
```
3. run RENET2 on parsed data
```
MODEL_DIR=${R2_DIR}/models/ft_models          # using the pretrained 10 models at ft_models
renet2 predict --raw_data_dir ${R2_DIR}/data/test_data/ --model_dir ${R2_DIR}/models/ft_models/ --gda_fn_d ${R2_DIR}/data/test_data/ --models_number 4 --batch_size 8 --max_doc_num 10 --no_cache_file 
```
Output data: predicted Gene-Disease Associations are stored in `${R2_DIR}/data/test_data/gda_rst.tsv`

### Example of running RENET2 Model on abstract data

to try run RENET2 on abstract, you can using the code as:
```
renet2 predict --raw_data_dir ${R2_DIR}/data/abs_data/2nd_ann/ \
--model_dir ${R2_DIR}/models/ \
--gda_fn_d ${R2_DIR}/data/test_data/ \
--models_number 1 \
--model_name Bst_abs_10 \
--batch_size 8 \
--no_cache_file \
--fix_snt_n 32 \
--file_name_ann anns.txt

# then go to benchmark folder and run the following to checked the trained models
python calculate_metrics_with_input.py ${R2_DIR}/data/abs_data/2nd_ann/labels.txt ${R2_DIR}/data/test_data/gda_rst.tsv

```


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

Modules in `renet2` are for model training/testing.

*For the Modules listed below, please use the `-h` or `--help` option for checking available options.*

`renet2` | renet2 program
---: | ---
`train` | Module for training RENET2 models.
`predict` | Using RENET2 models to predict gene-disease associations.
`evaluate_renet2_ft_cv` | Evaluating trained RENET2 models and using cross-validation.
`download_data` | Downloading articles from PMC/PTC with provided PMID/PMCID list. (please check example an RENET2/src/nb_scripts/pre_precoss/ for full-text dataset)
`parse_data` | Parsing articles from RENET2. (please check example an RENET2/src/nb_scripts/pre_precoss/ for full-text dataset)
`normalize_ann` | Normlize the annotation ID
`install_geniass` | Install geniass for parse_data module, if fail, please try `conda install ruby` to install ruby first


## Benchmark

### Run BeFree

```
pip install pymongo
pip install regex
cd benchmark/BeFree
git clone git@bitbucket.org:ibi_group/befree.git

wget http://www.bio8.cs.hku.hk/RENET2/renet2_bm_befree.tar.gz
tar -xf renet2_bm_befree.tar.gz
# get BeFree input
run Generate_BeFree_Input.ipynb on python jypyter notebook to genrate BeFree input
sh benchmark_befree.sh
```

### Run DTMiner
```
cd benchmark/DTMiner
wget http://www.bio8.cs.hku.hk/RENET2/renet2_bm_dtminer.tar.gz
tar -xf renet2_bm_dtminer.tar.gz
# get DTMiner input
run Generate_DTMiner_Input.ipynb on python jypyter notebook to genrate BeFree input
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

### Run RENET (on full-text)
```
cd benchmark

run Generate_RENET_Input.ipynb on python jypyter notebook
```

### Benchmark (Using RENET2 Cross-validation to Evalutate RENET2/BeFree/DTMiner/BioBERT Results)

```
renet2 evaluate_renet2_ft_cv --epochs 10 --raw_data_dir ${R2_DIR}/data/ft_data/ --annotation_info_dir ${R2_DIR}/data/ft_info/ --rst_file_prefix ft_base --have_SiDa ${R2_DIR}/data/ft_info/ft_base/ft_base --pretrained_model_p ${R2_DIR}/models/Bst_abs_10 --no_cache_file --use_cuda
```

### Benchmark RENET2 on Abstract Data

```
# training RENET2 model on abstract data
run ./src/nb_scripts/build_best_model_abs.ipynb on jupyter notebook
# Using cross-validation to benchmarking RENET2 on abstract data
run ./src/nb_scripts/exp_abs.ipynb on jupyter notebook
```


note that RENET2 can benchmark should be benchmark on abstract data via cross validation.




## (Optional) Visualization

Found and visualze a pair of gene-disease annotation obtrained from Pubtar Central.


```
run ./src/nb_scripts/vis_text.ipynb on jupyter notebook
```
![RENET2 vis](/repo_f/RENET2_vis.png "RENET2 vis")


