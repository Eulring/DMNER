
# Overview
```
- S1-EBD（Entity Boundary Detaction Module
    | - biobertNER (code for Flat Supervised NER)
    | - dsner (code for DSNER NER)
    | - uni (code for unified NER)

- S2-BEM (Biomedical Entity Matching Module)
    | - dictionary 
    | - script (script to run the BEM and DR)
    | - testdata (Stores the output data from the EBD model)
    | - output（Output results from evaluation scripts for ensemble results）
```




# Supervised NER

Downloading huggingface biobert-v1.1 embeding into S1-EBD/embed

Our flat NER module is based on [biobert-pytorch](https://github.com/dmis-lab/biobert-pytorch), so the requirements should be consistent with that project.

You can download the biomedical NER dataset following this [link](https://github.com/dmis-lab/biobert)

__(1) EBD__
```
export DMNER_ROOT=/home/test2/DMNER
cd $DMNER_ROOT/DMNER/S1-EBD/biobertNER/NER 
sh train_ncbi.sh $GPUID
sh infer_ncbi.sh $GPUID
```

__(2): BEM__

Get init dictionary
```
export DMNER_ROOT=/home/test2/DMNER
cd $DMNER_ROOT/S2-BEM/script
python dict_init_fromgold.py --dname NCBI --etype Disease --droot ${DMNER_ROOT}
```


Dictionary refinement & Ensemble the results
```
python all.py --dname NCBI --gpu $GPUID
```





# Distantly Supervised NER
The EBD backbone of DS-NER is borrowed from [MRC](https://github.com/ShannonAI/mrc-for-flat-nested-ner). The environment needs to be reconfigured.

The trusted entities and unknown entities used in training come from [autoner](https://github.com/shangjingbo1226/AutoNER).


__(1) EBD__
```
export DMNER_ROOT=/home/test2/DMNER
cd $DMNER_ROOT/S1-EBD/dsner
sh paral_train_bc5cdr.sh $GPUID
sh infer_bc5cdr.sh $GPUID
```


__(2) BEM__
Dictionary refinement & Ensemble the results
```
python all.py --dname BC5CDR --gpu $GPUID
```





# Unified NER

__(1) EBD__
```
export DMNER_ROOT=/home/test2/DMNER
cd $DMNER_ROOT/S1-EBD/uni
sh paral_train_uni.sh $GPUID
sh infer_bc5cdr.sh $GPUID
```


__(2) BEM__
Dictionary refinement & Ensemble the results
```
python all.py --dname BC5CDR-UNI --gpu $GPUID
```



