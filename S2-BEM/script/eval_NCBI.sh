MODEL_DIR="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
DICT_PATH=$2
DATA_DIR=../testdata/NCBI/test_iob.json

CUDA_VISIBLE_DEVICES=$1 python3 eval_new.py \
	--model_dir $MODEL_DIR \
	--dictionary_path $DICT_PATH \
	--data_dir $DATA_DIR \
	--use_cuda \
	--max_length 25 \
	--entity_types Disease