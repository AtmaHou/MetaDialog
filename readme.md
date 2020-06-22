# Meta Dialog Platform (MDP)
Code usage and other instructions.

## 1. Get Started
### Environment Requirement
```
python >= 3.6
pytorch >= 0.4.1
pytorch_pretrained_bert >= 0.6.1
allennlp >= 0.8.2
```

### Prepare pre-trained embedding:

#### BERT
Down the pytorch bert model, or convert tensorflow param yourself as follow:
```bash
export BERT_BASE_DIR=/users4/ythou/Projects/Resources/bert-base-uncased/uncased_L-12_H-768_A-12/

pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch
  $BERT_BASE_DIR/bert_model.ckpt
  $BERT_BASE_DIR/bert_config.json
  $BERT_BASE_DIR/pytorch_model.bin
```
Set BERT path in the `scripts/*.sh` whose variable name is `pretrained_model_path` 

### Prepare data
Original data is available by contacting me, or you can generate it:
Set test, train, dev data file path in ./scripts/

#### Few-shot Data Generation
We provide a generation tool for converting normal data into few-shot/meta-episode style. 

### Train and test the main model
Run command line:
```bash
source ./scripts/run_main.sh [gpu id split with ','] [data set name]
```

Example:
```bash
source ./scripts/run_main.sh 1,2 snips
```



