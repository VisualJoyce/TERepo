TERepo
===============

A Text Editing Repository for reproduction and innovation.

Preprocessing
-------------
The datasets used are publicly available online.

* MuCGEC 
* FCGEC

```shell
PYTHONPATH=src python3.9 preprocess/text_editing/mucgec_to_wds.py \
--input_dir data/datasets/MuCGEC --output_dir data/annotations/text_editing/zh/mucgec 
```

Training
--------

```shell
bash train_gector.sh gector_focal gec-zh-gector-bert-large 0 mucgec "--focal_gamma 2"
```


Predicting
----------

Submission
----------
