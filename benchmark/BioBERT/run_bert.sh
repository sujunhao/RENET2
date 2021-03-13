export BIOBERT_DIR=/autofs/bal31/jhsu/old/biobert-pretrained/biobert_v1.1_pubmed
export BIOBERT_B_D=/autofs/bal31/jhsu/home/git/biobert
export RE_DIR=./
export TASK_NAME=gad
export OUTPUT_DIR=./re_outputs_1
#python run_re.py --task_name=$TASK_NAME --do_train=true --do_eval=true --do_predict=true --vocab_file=$BIOBERT_DIR/vocab.txt --bert_config_file=$BIOBERT_DIR/bert_config.json --init_checkpoint=$BIOBERT_DIR/model.ckpt-1000000 --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3.0 --do_lower_case=false --data_dir=$RE_DIR --output_dir=$OUTPUT_DIR
python $BIOBERT_B_D/run_re.py --task_name=$TASK_NAME --do_train=false --do_eval=false --do_predict=true --vocab_file=$BIOBERT_DIR/vocab.txt --bert_config_file=$BIOBERT_DIR/bert_config.json --init_checkpoint=$BIOBERT_DIR/model.ckpt-1000000 --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3.0 --do_lower_case=false --data_dir=$RE_DIR --output_dir=$OUTPUT_DIR
