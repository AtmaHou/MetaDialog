# Meta Dialog Platform (MDP)

Meta Dialog Platform, is a tool for Few-Shot Learning. The platform now can be used for `classification` and `sequence labeling` and `joint learning`(which only shares embedding).

As [Electra](https://openreview.net/forum?id=r1xMH1BtvB) proposed, we can use `Electra small model` to speed up our exploitation because it's so small. And if you want to get the chinese version, you can go [Chinese-Electra](https://github.com/ymcui/Chinese-ELECTRA).


## Get Started

### Environment Requirement
```
python>=3.6
torch>=1.2.0
transformers>=2.9.0
numpy>=1.17.0
tqdm>=4.31.1
allennlp>=0.8.4
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

The generation tool is in the `scripts/other_tool` folder named `meta_dataset_generator`.
And we provide a bash script named `gen_meta_data.sh` which is used to control to run the generation tool.

Run command line:
```bash
cd scripts
source ./gen_meta_data.sh
```

##### gen_meta_data.sh

There are some params for you to control the generation process:
- `input_dir`: raw data path
- `output_dir`: output data path
- `episode_num`: the number of episode which you want to generate
- `support_shots_lst`: to specified the support shot size in each episode, we can specified multiple number to generate at the same time.
- `query_shot`: to specified the query shot size in each episode
- `seed_lst`: random seed list to control random generation
- `use_fix_support`:  set the fix support in dev dataset
- `dataset_lst`: specified the dataset type which our tool can handle, there are some choices: `stanford` & `SLU` & `TourSG` & `SMP`. 

> If you want to handle other type of dataset, 
> you can add your code for processing dataset in `meta_dataset_generator/raw_data_loader.py`, 
> and add the dataset type to parameter controller in `meta_dataset_generator/generate_meta_dataset.py`.

##### few-shot/meta-episode style

```
{
  "domain_name": [
    {  // episode
      "support": {  // support set
        "seq_ins": [["we", "are", "friends", "."], ["how", "are", "you", "?"]],  // input sequence
        "seq_outs": [["O", "O", "O", "O"], ["O", "O", "O", "O"]],  // output sequence in sequence labeling task
        "labels": [["statement"], ["query"]]  // output labels in classification task
      },
      "query": {  // query set
        "seq_ins": [["we", "are", "friends", "."], ["how", "are", "you", "?"]],
        "seq_outs": [["O", "O", "O", "O"], ["O", "O", "O", "O"]],
        "labels": [["statement"], ["query"]]
      }
    },
    ...
  ],
  ...
}

```


### Train and test the main model

Run command line:
```bash
source ./scripts/run_main.sh [gpu id split with ','] [data set name]
```

Example:
```bash
source ./scripts/run_main.sh 1,2 snips
```

#### bash scripts

We provide some bash scripts for convenience. As mentioned, we provide two scripts for each task respectively, each task has the `Electra` version and `BERT` version.

- classification
    - `run_electra_sc.sh`
    - `run_bert_sc.sh`
- sequence labeling
    - `run_electra_sl.sh`
    - `run_bert_sl.sh`
- joint learning
    - `run_electra_sc+sl.sh`
    - `run_bert_sc+sl.sh`

#### parameters

There are many parameters to control the train & test process, but there are some main parameters you should change and know.
- `do_debug`: set for debug model
- `task`: specify the target task which is a list split with space(should with backslash in bash script)
- `emission`: specify the emission choice which is a list match with task list, others is same with `task`
- dataset name: `support_shots_lst` & `query_shot` & `episode` & `cross_id` are used for specify the dataset path
- `embedder`: specify the embedder type, the choices are `bert` & `electra` & `sep_bert` & `glove`, in which `sep_bert` are not pair-wise embedding while others do.
- `pretrained_model_path`: the pre-trained model path
- `pretrained_vocab_path`: the vocabulary of pre-trained model, you can specify the file or its parent folder (default find the `vocab.txt` in it)
- `base_data_dir`: the path of your dataset, the `base` mean that in which there are sub-folders (the sub-folders is named with `dataset name` mentioned at second item). If not, you can adjust the script.


## Information

The platform is developed by [HIT-SCIR](http://ir.hit.edu.cn/). If you have any question and advice for it, please contact us(Yutai Hou - [ythou@ir.hit.edu.cn](mailto:ythou@ir.hit.edu.cn) or Yongkui Lai - [yklai@ir.hit.edu.cn](mailto:yklai@ir.hit.edu.cn)).
