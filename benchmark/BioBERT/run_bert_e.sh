export BIOBERT_DIR=/autofs/bal31/jhsu/old/biobert-pretrained/biobert_v1.1_pubmed
#export BIOBERT_B_D=/autofs/bal31/jhsu/home/git/biobert/biocodes/
export BIOBERT_B_D=.
export RE_DIR=./
export TASK_NAME=gad
export OUTPUT_DIR=./re_outputs_1
python $BIOBERT_B_D/re_eval.py --output_path=$OUTPUT_DIR/test_results.tsv --answer_path=$RE_DIR/test.tsv
