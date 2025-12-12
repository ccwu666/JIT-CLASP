# JIT-CLASP-replication-package

### Environment Settings

```
https://github.com/ccwu666/JIT-CLASP.git
conda create --name jitclasp python=3.8
conda activate jitclasp
pip install torch==1.9.0+cu111  -f ``https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

### Data and Pre-trained Model Files Preparation

1. Extract the data used to train and evaluate JIT-CLASP via ```unzip data/lapredict_unfied.zip```.

2. Manually download the **added_tokens.json, config.json, merges.txt, pytorch_model.bin, special_tokens_map.json, tokenizer_config.json, and vocab.json** from (https://huggingface.co/microsoft/codebert-base/tree/main), upload them to the **microsoft/codebert-base** folder.


### Run JIT-CLASP

```
python run_lapredict_scl_cross_attention.py
```

