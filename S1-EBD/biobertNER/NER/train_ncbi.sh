export SAVE_DIR=./output
export DATA_DIR=../datasets

export MAX_LENGTH=192
export BATCH_SIZE=8
export NUM_EPOCHS=20
export SAVE_STEPS=4000
export ENTITY=NCBI
export SEED=1

CUDA_VISIBLE_DEVICES=$1 python run_ner.py \
    --data_dir ${DATA_DIR}/${ENTITY}/ \
    --labels ${DATA_DIR}/${ENTITY}/labels.txt \
    --model_name_or_path dmis-lab/biobert-base-cased-v1.1 \
    --output_dir ${SAVE_DIR}/${ENTITY} \
    --max_seq_length ${MAX_LENGTH} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --save_steps ${SAVE_STEPS} \
    --seed ${SEED} \
    --do_train \
    --do_eval \
    --do_predict \
    --overwrite_output_dir