#!/usr/bin/env bash

echo usage: pass dataset list as param, split with space
echo eg: source gen_mate_data.sh atis

dataset_lst=(smp)
#dataset_lst=(toursg)
#dataset_lst=(stanford)
#dataset_lst=(atis stanford toursg)

# ======= size setting ======
support_shots_lst=(1)
#support_shots_lst=(5)
episode_num=50  # We could over generation and select part of for each epoch
query_shot=4
word_piece_data=True
way=-1

remove_rate=80


# ====== general setting =====
seed_lst=(0)

#task=sc
#task=sl
task=slu

#dup_query=--dup_query  # dup_query set empty to not allow duplication between query and support
dup_query=

allow_override=--allow_override

check=--check

# ====== train & test setting ======
split_basis=domain

#eval_confif_id_lst=(1)  # for snips
#eval_config_id_lst=(0 1 2 3 4 5)  # for toursg
label_type_lst=(attribute)

#use_fix_support=
use_fix_support=--use_fix_support

# ======= default path (for quick distribution) ==========
#input_dir=/Users/lyk/Work/Dialogue/FewShot/SMP正式数据集/
#output_dir=/Users/lyk/Work/Dialogue/FewShot/SMP正式数据集/SmpMetaData/
input_dir=/Users/lyk/Work/Dialogue/FewShot/SMP/SMP_Final_Origin2
output_dir=/Users/lyk/Work/Dialogue/FewShot/SMP/smp_final/

now=$(date +%s)

echo \[START\] set jobs on dataset \[ ${dataset_lst[@]} \]
# === Loop for all case and run ===
for seed in ${seed_lst[@]}
do
  for dataset in ${dataset_lst[@]}
  do
    for support_shots in ${support_shots_lst[@]}
    do
      echo \[CLI\] generate with \[ ${use_fix_support} \]
      input_path=${input_dir}
      mark=try

      export OMP_NUM_THREADS=2  # threads num for each task
      python3 ./other_tool/meta_dataset_generator/generate_meta_dataset.py \
        --input_path ${input_path} \
        --output_dir ${output_dir} \
        --dataset ${dataset} \
        --episode_num ${episode_num} \
        --support_shots ${support_shots} \
        --query_shot ${query_shot} \
        --way ${way} \
        --task ${task} \
        --seed ${seed} \
        --split_basis ${split_basis} \
        --remove_rate ${remove_rate} \
        ${use_fix_support} \
        --mark ${mark} ${dup_query} ${allow_override} ${check} > ${output_dir}${dataset}.spt_s_${support_shots}.q_s_${query_shot}.ep_${episode_num}${use_fix_support}-${now}.log
    done
  done
done





