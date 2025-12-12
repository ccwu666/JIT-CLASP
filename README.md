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


### Run DL and ML

```
python run_lapredict_scl_cross_attention.py
python ./Sim/run_ml_5seeds.py
```

**Note:** This step can generate the pred_scores of the DL and ML, you can find in /model and ./Sim/pred_scores folder.

### Combine Pred_scores of DL and ML

1.Find the pred_scores of the DL and ML. 
2.Modify the com_path and sim_path of the quick_run_combine.py
3.run the follow:
```
python ./combine/quick_run_combine.py
```

### You can get the final results of the JIT-CLASP after finishing above steps

