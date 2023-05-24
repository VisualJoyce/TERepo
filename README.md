![TERepo](https://github.com/VisualJoyce/TERepo/assets/2136700/74ce4d48-7aa4-4a69-9ffe-3025ae24bf9f)

TERepo
===============

A Text Editing Repository for reproduction and innovation.

GEC
---
This repo contains code for the following paper accepted to ACL 2023 Findings
```bibtex
@inproceedings{tan-etal-2023-focal,
    title = "Focal Training and Tagger Decouple for Grammatical Error Correction",
    author = "Minghuan, Tan  and
      Min, Yang  and
      Ruifeng, Xu",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = july,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    abstract = "In this paper, we investigate how to improve tagging-based Grammatical Error Correction models. We address two issues of current tagging-based approaches, label imbalance issue, and tagging entanglement issue. Then we propose to down-weight the loss of correctly classified labels using Focal Loss and decouple the error detection layer from the label tagging layer through an extra self-attention-based matching module. Experiments on three recent Chinese Grammatical Error Correction datasets show that our proposed methods are effective. We further analyze choices of hyper-parameters for Focal Loss and inference tweaking.",
}
```

### Preprocessing

The datasets used are publicly available online.

* MuCGEC 
* FCGEC
* MCSCSet

```shell
PYTHONPATH=src python3.9 preprocess/text_editing/mucgec_to_wds.py \
--input_dir data/datasets/MuCGEC --output_dir data/annotations/text_editing/zh/mucgec 
```

### Training

```shell
bash examples/gector/train.sh gector_focal aihijo/gec-zh-gector-bert-large 0 mucgec "--focal_gamma=2"
bash examples/gector/train.sh gector_focal aihijo/gec-zh-gector-bert-large 0 fcgec "--focal_gamma=2"
bash examples/gector/train.sh gector_focal aihijo/gec-zh-gector-bert-large 0 mcscset "--focal_gamma=2"
```

### Predicting

### Submission
