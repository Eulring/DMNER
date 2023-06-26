Example of dictionary refinement algorithm:

CUDA_VISIBLE_DEVICES=1 python3 dict_fromdev-5-22.py \
	--fpath_dev ../testdata/BC5CDR-chem/dev_iob.json \
    --target_file  BC5CDR-chem/dict-dev-iob-t2.txt \
    --iter 20 \
    --thres 2 \
    --shuffle_seed 1 \
    --sample_rate 1 \
    --ava_entity Chemical \
    --ner_type iob \
    --dname BC5CDR-chem

% 