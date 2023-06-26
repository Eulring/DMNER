import os
import argparse

parser = argparse.ArgumentParser(description='pipeline')
parser.add_argument('--dname', type=str, default='JNLPBA')
parser.add_argument('--gpu', type=str, default='1')
args = parser.parse_args()

dname = args.dname




if dname in ['BC2GM', 'JNLPBA']:
    os.system("CUDA_VISIBLE_DEVICES=## python3 dict_fromdev.py --fpath_dev ../testdata/@/dev_iob.json     --target_file  @/dict-dev-iob-1.txt     --iter 20     --drop_rate 0     --select_rate 0     --shuffle_seed 1     --sample_rate 1     --ava_entity Gene     --ner_type iob     --dname @".replace('##', args.gpu).replace('@', dname))
    os.system("sh eval_@.sh ## ../dictionary/@/dict-dev-iob-1.txt".replace('##', args.gpu).replace('@', dname))

    os.system("CUDA_VISIBLE_DEVICES=## python3 dict_fromdev.py --fpath_dev ../testdata/@/dev_iob.json     --target_file  @/dict-dev-iob-2.txt     --iter 20     --drop_rate 0     --select_rate 0     --shuffle_seed 2     --sample_rate 1     --ava_entity Gene     --ner_type iob     --dname @".replace('##', args.gpu).replace('@', dname))
    os.system("sh eval_@.sh ## ../dictionary/@/dict-dev-iob-2.txt".replace('##', args.gpu).replace('@', dname))

    os.system("CUDA_VISIBLE_DEVICES=## python3 dict_fromdev.py --fpath_dev ../testdata/@/dev_iob.json     --target_file  @/dict-dev-iob-3.txt     --iter 20     --drop_rate 0     --select_rate 0     --shuffle_seed 3     --sample_rate 1     --ava_entity Gene     --ner_type iob     --dname @".replace('##', args.gpu).replace('@', dname))
    os.system("sh eval_@.sh ## ../dictionary/@/dict-dev-iob-3.txt".replace('##', args.gpu).replace('@', dname))

    os.system('python assemble.py --dname @'.replace('@', dname))


if dname in ['NCBI', 'BC5CDR-disease']:

    os.system("CUDA_VISIBLE_DEVICES=## python3 dict_fromdev.py --fpath_dev ../testdata/@/dev_iob.json     --target_file  @/dict-dev-iob-1.txt     --iter 20     --drop_rate 0     --select_rate 0     --shuffle_seed 1     --sample_rate 1     --ava_entity Disease     --ner_type iob     --dname @".replace('##', args.gpu).replace('@', dname))
    os.system("sh eval_@.sh ## ../dictionary/@/dict-dev-iob-1.txt".replace('##', args.gpu).replace('@', dname))

    os.system("CUDA_VISIBLE_DEVICES=## python3 dict_fromdev.py --fpath_dev ../testdata/@/dev_iob.json     --target_file  @/dict-dev-iob-2.txt     --iter 20     --drop_rate 0     --select_rate 0     --shuffle_seed 2     --sample_rate 1     --ava_entity Disease     --ner_type iob     --dname @".replace('##', args.gpu).replace('@', dname))
    os.system("sh eval_@.sh ## ../dictionary/@/dict-dev-iob-2.txt".replace('##', args.gpu).replace('@', dname))

    os.system("CUDA_VISIBLE_DEVICES=## python3 dict_fromdev.py --fpath_dev ../testdata/@/dev_iob.json     --target_file  @/dict-dev-iob-3.txt     --iter 20     --drop_rate 0     --select_rate 0     --shuffle_seed 13     --sample_rate 1     --ava_entity Disease     --ner_type iob     --dname @".replace('##', args.gpu).replace('@', dname))
    os.system("sh eval_@.sh ## ../dictionary/@/dict-dev-iob-3.txt".replace('##', args.gpu).replace('@', dname))

    os.system('python assemble.py --dname @'.replace('@', dname))


if dname in ['BC4CHEMD', 'BC5CDR-chem']:
    os.system("CUDA_VISIBLE_DEVICES=## python3 dict_fromdev.py --fpath_dev ../testdata/@/dev_iob.json     --target_file  @/dict-dev-iob-1.txt     --iter 20     --drop_rate 0     --select_rate 0     --shuffle_seed 1     --sample_rate 1     --ava_entity Chemical     --ner_type iob     --dname @".replace('##', args.gpu).replace('@', dname))
    os.system("sh eval_@.sh ## ../dictionary/@/dict-dev-iob-1.txt".replace('##', args.gpu).replace('@', dname))

    os.system("CUDA_VISIBLE_DEVICES=## python3 dict_fromdev.py --fpath_dev ../testdata/@/dev_iob.json     --target_file  @/dict-dev-iob-2.txt     --iter 20     --drop_rate 0     --select_rate 0     --shuffle_seed 2     --sample_rate 1     --ava_entity Chemical     --ner_type iob     --dname @".replace('##', args.gpu).replace('@', dname))
    os.system("sh eval_@.sh ## ../dictionary/@/dict-dev-iob-2.txt".replace('##', args.gpu).replace('@', dname))

    os.system("CUDA_VISIBLE_DEVICES=## python3 dict_fromdev.py --fpath_dev ../testdata/@/dev_iob.json     --target_file  @/dict-dev-iob-3.txt     --iter 20     --drop_rate 0     --select_rate 0     --shuffle_seed 3     --sample_rate 1     --ava_entity Chemical     --ner_type iob     --dname @".replace('##', args.gpu).replace('@', dname))
    os.system("sh eval_@.sh ## ../dictionary/@/dict-dev-iob-3.txt".replace('##', args.gpu).replace('@', dname))

    os.system('python assemble.py --dname @'.replace('@', dname))

if dname in ['BC5CDR']:

    os.system("CUDA_VISIBLE_DEVICES=## python3 dict_fromdev.py --fpath_dev ../testdata/@/dev_mask.json     --target_file  @/dict-dev-mask-1.txt     --iter 20     --drop_rate 0     --select_rate 0     --shuffle_seed 1     --sample_rate 1     --ava_entity Chemical*Disease     --ner_type nest     --dname @".replace('##', args.gpu).replace('@', dname))
    os.system("sh eval_@.sh ## ../dictionary/@/dict-dev-mask-1.txt".replace('##', args.gpu).replace('@', dname))

    os.system("CUDA_VISIBLE_DEVICES=## python3 dict_fromdev.py --fpath_dev ../testdata/@/dev_mask.json     --target_file  @/dict-dev-mask-2.txt     --iter 20     --drop_rate 0     --select_rate 0     --shuffle_seed 2     --sample_rate 1     --ava_entity Chemical*Disease     --ner_type nest     --dname @".replace('##', args.gpu).replace('@', dname))
    os.system("sh eval_@.sh ## ../dictionary/@/dict-dev-mask-2.txt".replace('##', args.gpu).replace('@', dname))

    os.system("CUDA_VISIBLE_DEVICES=## python3 dict_fromdev.py --fpath_dev ../testdata/@/dev_mask.json     --target_file  @/dict-dev-mask-3.txt     --iter 20     --drop_rate 0     --select_rate 0     --shuffle_seed 3     --sample_rate 1     --ava_entity Chemical*Disease     --ner_type nest     --dname @".replace('##', args.gpu).replace('@', dname))
    os.system("sh eval_@.sh ## ../dictionary/@/dict-dev-mask-3.txt".replace('##', args.gpu).replace('@', dname))

    os.system('python assemble.py --dname @'.replace('@', dname))


if dname in ['BC5CDR-UNI']:

    os.system("CUDA_VISIBLE_DEVICES=## python3 dict_fromdev.py --fpath_dev ../testdata/@/dev_uni.json     --target_file  @/dict-dev-uni-1.txt     --iter 20    --shuffle_seed 1     --sample_rate 1     --ava_entity Chemical*Disease     --ner_type nest     --dname @".replace('##', args.gpu).replace('@', dname))
    os.system("CUDA_VISIBLE_DEVICES=## python3 dict_fromdev.py --fpath_dev ../testdata/@/dev_uni.json     --target_file  @/dict-dev-uni-2.txt     --iter 20    --shuffle_seed 2     --sample_rate 1     --ava_entity Chemical*Disease     --ner_type nest     --dname @".replace('##', args.gpu).replace('@', dname))
    os.system("CUDA_VISIBLE_DEVICES=## python3 dict_fromdev.py --fpath_dev ../testdata/@/dev_uni.json     --target_file  @/dict-dev-uni-3.txt     --iter 20    --shuffle_seed 3     --sample_rate 1     --ava_entity Chemical*Disease     --ner_type nest     --dname @".replace('##', args.gpu).replace('@', dname))
    os.system("sh eval_@.sh ## ../dictionary/@/dict-dev-uni-1.txt".replace('##', args.gpu).replace('@', dname))
    os.system("sh eval_@.sh ## ../dictionary/@/dict-dev-uni-2.txt".replace('##', args.gpu).replace('@', dname))
    os.system("sh eval_@.sh ## ../dictionary/@/dict-dev-uni-3.txt".replace('##', args.gpu).replace('@', dname))

    os.system('python assemble.py --dname @'.replace('@', dname))




if dname == 's800':
    os.system("CUDA_VISIBLE_DEVICES=## python3 dict_fromdev.py --fpath_dev ../testdata/@/dev_iob.json     --target_file  @/dict-dev-iob-1.txt     --iter 20     --shuffle_seed 1     --sample_rate 1     --ava_entity Species     --ner_type iob     --dname @".replace('##', args.gpu).replace('@', dname))
    os.system("CUDA_VISIBLE_DEVICES=## python3 dict_fromdev.py --fpath_dev ../testdata/@/dev_iob.json     --target_file  @/dict-dev-iob-2.txt     --iter 20     --shuffle_seed 2     --sample_rate 1     --ava_entity Species     --ner_type iob     --dname @".replace('##', args.gpu).replace('@', dname))
    os.system("CUDA_VISIBLE_DEVICES=## python3 dict_fromdev.py --fpath_dev ../testdata/@/dev_iob.json     --target_file  @/dict-dev-iob-3.txt     --iter 20     --shuffle_seed 3     --sample_rate 1     --ava_entity Species     --ner_type iob     --dname @".replace('##', args.gpu).replace('@', dname))
    os.system("sh eval_@.sh ## ../dictionary/@/dict-dev-iob-1.txt".replace('##', args.gpu).replace('@', dname))
    os.system("sh eval_@.sh ## ../dictionary/@/dict-dev-iob-2.txt".replace('##', args.gpu).replace('@', dname))
    os.system("sh eval_@.sh ## ../dictionary/@/dict-dev-iob-3.txt".replace('##', args.gpu).replace('@', dname))

    os.system('python assemble.py --dname @'.replace('@', dname))


if dname == 'linnaeus':
    os.system("CUDA_VISIBLE_DEVICES=## python3 dict_fromdev.py --fpath_dev ../testdata/linnaeus/dev_iob.json     --target_file  linnaeus/dict-dev-iob-1.txt     --iter 20     --drop_rate 0     --select_rate 0     --shuffle_seed 1     --sample_rate 1     --ava_entity Species     --ner_type iob     --dname linnaeus".replace('##', args.gpu))
    os.system("sh eval_linnaeus.sh ## ../dictionary/linnaeus/dict-dev-iob-1.txt".replace('##', args.gpu))

    os.system("CUDA_VISIBLE_DEVICES=## python3 dict_fromdev.py --fpath_dev ../testdata/linnaeus/dev_iob.json     --target_file  linnaeus/dict-dev-iob-2.txt     --iter 20     --drop_rate 0     --select_rate 0     --shuffle_seed 2     --sample_rate 1     --ava_entity Species     --ner_type iob     --dname linnaeus".replace('##', args.gpu))
    os.system("sh eval_linnaeus.sh ## ../dictionary/linnaeus/dict-dev-iob-2.txt".replace('##', args.gpu))

    os.system("CUDA_VISIBLE_DEVICES=## python3 dict_fromdev.py --fpath_dev ../testdata/linnaeus/dev_iob.json     --target_file  linnaeus/dict-dev-iob-3.txt     --iter 20     --drop_rate 0     --select_rate 0     --shuffle_seed 3     --sample_rate 1     --ava_entity Species     --ner_type iob     --dname linnaeus".replace('##', args.gpu))
    os.system("sh eval_linnaeus.sh ## ../dictionary/linnaeus/dict-dev-iob-3.txt".replace('##', args.gpu))

    os.system('python assemble.py --dname linnaeus')


# if dname == 'NCBI':
#     os.system("CUDA_VISIBLE_DEVICES=## python3 dict_fromdev.py --fpath_dev ../testdata/NCBI/dev_iob.json     --target_file  NCBI/dict-dev-iob-1.txt     --iter 10     --drop_rate 0     --select_rate 0     --shuffle_seed 1     --sample_rate 1     --ava_entity Disease     --ner_type iob     --dname NCBI".replace('##', args.gpu))
#     os.system("sh eval_ncbi.sh ## ../dictionary/NCBI/dict-dev-iob-1.txt".replace('##', args.gpu))

#     os.system("CUDA_VISIBLE_DEVICES=## python3 dict_fromdev.py --fpath_dev ../testdata/NCBI/dev_iob.json     --target_file  NCBI/dict-dev-iob-2.txt     --iter 10     --drop_rate 0     --select_rate 0     --shuffle_seed 2     --sample_rate 1     --ava_entity Disease     --ner_type iob     --dname NCBI".replace('##', args.gpu))
#     os.system("sh eval_ncbi.sh ## ../dictionary/NCBI/dict-dev-iob-2.txt".replace('##', args.gpu))

#     os.system("CUDA_VISIBLE_DEVICES=## python3 dict_fromdev.py --fpath_dev ../testdata/NCBI/dev_iob.json     --target_file  NCBI/dict-dev-iob-3.txt     --iter 10     --drop_rate 0     --select_rate 0     --shuffle_seed 3     --sample_rate 1     --ava_entity Disease     --ner_type iob     --dname NCBI".replace('##', args.gpu))
#     os.system("sh eval_ncbi.sh ## ../dictionary/NCBI/dict-dev-iob-3.txt".replace('##', args.gpu))


#     os.system('python assemble.py --dname NCBI')