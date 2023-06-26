
export SAVE_DIR=$DMNER_ROOT/S1-EBD/biobertNER/NER/output
export DATA_DIR=$DMNER_ROOT/S1-EBD/biobertNER/datasets

export MAX_LENGTH=192
export BATCH_SIZE=8
export NUM_EPOCHS=10
export SAVE_STEPS=200
export ENTITY=NCBI
export SEED=1

CUDA_VISIBLE_DEVICES=$1 python run_ner.py \
    --data_dir ${DATA_DIR}/${ENTITY}/ \
    --labels ${DATA_DIR}/${ENTITY}/labels.txt \
    --model_name_or_path ${SAVE_DIR}/${ENTITY} \
    --output_dir ${SAVE_DIR}/${ENTITY} \
    --max_seq_length ${MAX_LENGTH} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --save_steps ${SAVE_STEPS} \
    --seed ${SEED} \
    --do_predict \
    --overwrite_output_dir

python pred2bel.py \
    --dname NCBI \
    --etype Disease \
    --droot ${DMNER_ROOT}