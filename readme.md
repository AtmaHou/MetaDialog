# Meta Dialog Platform (MDP)

Meta Dialog Platform: a toolkit platform for **NLP Few-Shot Learning** tasks of:
- Text Classification
- Sequence Labeling

It also provides the baselines for:
- [Track-1 of SMP2020: Few-shot dialog language understanding](https://smp2020.aconf.cn/smp.html#3).
- [Benchmark Paper: "FewJoint: A Few-shot Learning Benchmark for Joint Language Understanding"]("https://arxiv.org/abs/2009.08138")

### Updates

- Updates 2020.9.17: FewJoint benchmark (Dataset for SMP) is available: [paper](https://arxiv.org/abs/2009.08138), [data](https://atmahou.github.io/attachments/FewJoint.zip), [reformatted data (for MetaDialog)](https://atmahou.github.io/attachments/FewJoint_for_MetaDialog.zip)

### Features
State-of-the-art solutions for Few-shot NLP:
-  Support Few-shot Learning for sequence-labeling task with state-of-the-art methods: CDT [(Hou et al., 2020)](https://arxiv.org/abs/2006.05702).
-  Support to use semantic within label name or label description. 
-  Support various deep pre-trained embedding compatible with [huggingface/transformers](https://github.com/huggingface/transformers), such as **[BERT](https://arxiv.org/abs/1810.04805)** and **[Electra](https://openreview.net/forum?id=r1xMH1BtvB)**.
-  Support pair-wise embedding mechanism ([Hou et al., 2020](https://arxiv.org/abs/2006.05702), [Gao et al., 2019](https://www.aclweb.org/anthology/D19-1649)).


Easy-to-start & flexible framework:
-  Provide tools for easy training & testing.
-  Support various few-shot models with unified and extendable interfaces, such as ProtoNet and TapNet.
-  Support easy-to-switch similarity-metrics and logits-scaling methods.
-  Provide tools of generating episode-style data for meta-learning.

## Citation
Please cite code and data:
```
@article{hou2020fewjoint,
	title={FewJoint: A Few-shot Learning Benchmark for Joint Language Understanding},
	author={Yutai Hou, Jiafeng Mao, Yongkui Lai, Cheng Chen, Wanxiang Che, Zhigang Chen, Ting Liu},
	journal={arXiv preprint},
	year={2020}
}
```


## Get Started

### Environment Requirement
```
python>=3.6
torch>=1.2.0
transformers>=2.9.0
numpy>=1.17.0
tqdm>=4.31.1
allennlp>=0.8.4
pytorch-nlp
```

### Example for Sequence Labeling
Here, we take the few-shot slot tagging and NER task from [(Hou et al., 2020)](https://arxiv.org/abs/2006.05702) as quick start examples.

#### Step1: Prepare pre-trained embedding
- Download the pytorch bert model, or convert tensorflow param by yourself with [scripts](https://github.com/huggingface/transformers/blob/master/src/transformers/convert_bert_original_tf_checkpoint_to_pytorch.py).
- Set BERT path in the `./scripts/run_1_shot_slot_tagging.sh` to your setting:
```bash
bert_base_uncased=/your_dir/uncased_L-12_H-768_A-12/
bert_base_uncased_vocab=/your_dir/uncased_L-12_H-768_A-12/vocab.txt
```

#### Step2: Prepare data
- Download the **compatible** few-shot data at here: [download](https://atmahou.github.io/attachments/new_FewShotNLU_data(ACL20).zip)

- Set test, train, dev data file path in `./scripts/run_1_shot_slot_tagging.sh` to your setting.
  
> For simplicity, your only need to set the root path for data as follow:
```bash
base_data_dir=/your_dir/ACL2020data/
```

#### Step3: Train and test the main model
- Build a folder to collect running log
```bash
mkdir result
```

- Execute cross-evaluation script with two params: -[gpu id] -[dataset name]

##### Example for 1-shot slot tagging:
```bash
source ./scripts/run_1_shot_slot_tagging.sh 0 snips
```  

##### Example for 1-shot NER:
```bash
source ./scripts/run_1_shot_slot_tagging.sh 0 ner
```

> To run 5-shots experiments, use `./scripts/run_5_shot_slot_tagging.sh`

### Other detailed functions and options:
You can experiment freely by passing parameters to `main.py` to choose different model architectures, hyperparameters, etc.

To view detailed options and corresponding descriptions, run commandline: 
```bash
python main.py --h
```

We provide scripts for general few-shot classification and sequence labeling task respectively:

- classification
    - `run_electra_sc.sh`
    - `run_bert_sc.sh`
- sequence labeling
    - `run_electra_sl.sh`
    - `run_bert_sl.sh`

The usage of these scripts are similar to process in Get Started.


## Run with FewJoint/SMP data
- Get reformatted FewJoint data at [here](https://atmahou.github.io/attachments/FewJoint_for_MetaDialog.zip) or construct episode-style data by yourself with [our tool](https://github.com/AtmaHou/MetaDialog#few-shot-data-construction-tool).
- Use script `./scripts/run_smp_bert_sc.sh` and `./scripts/run_smp_bert_sl.sh` to perform few-shot intent detection or few-shot slot filling respectively.
- Notice that: 
    1. Change train/dev/test path in the scripts before running. 
    2. Find predicted results at `trained_model_path` within running scripts.


## Few-shot Data Construction Tool
We also provide a generation tool for converting normal data into few-shot/meta-episode style. 
The tool is included at path: `scripts/other_tool/meta_dataset_generator.py`. 

Run following commandline to view detailed interface:
```bash
python generate_meta_dataset.py --h
```

For simplicity, we provide an example script to help generate few-shot data: `./scripts/gen_meta_data.sh`.

The following are some key params for you to control the generation process:
- `input_dir`: raw data path
- `output_dir`: output data path
- `episode_num`: the number of episode which you want to generate
- `support_shots_lst`: to specified the support shot size in each episode, we can specified multiple number to generate at the same time.
- `query_shot`: to specified the query shot size in each episode
- `seed_lst`: random seed list to control random generation
- `use_fix_support`:  set the fix support in dev dataset
- `dataset_lst`: specified the dataset type which our tool can handle, there are some choices: `stanford` & `SLU` & `TourSG` & `SMP`. 

> If you want to handle other type of dataset, 
> you can add your code for load raw dataset in `meta_dataset_generator/raw_data_loader.py`.


##### few-shot/meta-episode style data example

```json
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



## Acknowledgment

The platform is developed by [HIT-SCIR](http://ir.hit.edu.cn/). If you have any question and advice for it, please contact us(Yutai Hou - [ythou@ir.hit.edu.cn]() or Yongkui Lai - [yklai@ir.hit.edu.cn]()).
