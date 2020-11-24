#!/usr/bin/env bash
echo usage: pass gpu id list as param, split with ,
echo eg: source run_bert_siamese.sh 3,4 stanford

gpu_list=$1

# Comment one of follow 2 to switch debugging status
#do_debug=--do_debug
do_debug=

#restore=--restore_cpt
restore=

task=sc\ sl
#task=sl

use_schema=--use_schema
#use_schema=


# ======= dataset setting ======
dataset_lst=($2 $3)
support_shots_lst=(3)

query_shot=4
episode=20

cross_data_id=0  # for smp

# ====== train & test setting ======
#seed_lst=(0)
seed_lst=(6150 6151 6152)

#lr_lst=(0.000001 0.000005 0.00005)
lr_lst=(0.00001)

clip_grad=5

decay_lr_lst=(0.5)
#decay_lr_lst=(-1)

#upper_lr_lst=( 0.5 0.01 0.005)
#upper_lr_lst=(0.01)
upper_lr_lst=(0.001)
#upper_lr_lst=(0.0001)
#upper_lr_lst=(0.0005)
#upper_lr_lst=(0.1)
#upper_lr_lst=(0.001 0.1)

#fix_embd_epoch_lst=(1)
fix_embd_epoch_lst=(-1)
#fix_embd_epoch_lst=(1 2)

warmup_epoch=2

train_batch_size_lst=(4)
test_batch_size=2
#grad_acc=2
grad_acc=4
epoch=3

judge_joint_success=--judge_joint_success
#judge_joint_success=

# ==== model setting =========
# ---- encoder setting -----

#embedder=electra
embedder=bert
#embedder=sep_bert


# --------- emission setting --------
emission=proto_with_label\ tapnet # match with task


#similarity=cosine
#similarity=l2
similarity=dot

emission_normalizer=none
#emission_normalizer=softmax
#emission_normalizer=norm

#emission_scaler=none
#emission_scaler=fix
emission_scaler=learn
#emission_scaler=relu
#emission_scaler=exp


do_div_emission=-dbt
#do_div_emission=

ems_scale_rate_lst=(0.01)
#ems_scale_rate_lst=(0.01 0.02 0.05 0.005)

label_reps=sep
#label_reps=cat

ple_normalizer=none
ple_scaler=fix
#ple_scale_r=0.5
ple_scale_r_lst=(0.5)
#ple_scale_r=1
#ple_scale_r=0.01

tap_random_init=--tap_random_init
tap_random_init_r=0.5
tap_proto=--tap_proto
tap_proto_r=0.3
tap_mlp=
emb_log=

# ------ decoder setting -------
#decoder_lst=(rule)
#decoder_lst=(sms)
decoder_lst=(crf)
#decoder_lst=(crf sms)


#trans_init_lst=(fix rand)
trans_init_lst=(rand)

mask_trans=-mk_tr
#mask_trans=

trans_scaler=fix
#trans_scale_rate_lst=(10)
trans_scale_rate_lst=(1)

trans_rate=1
#trans_rate=0.8

trans_normalizer=none
#trans_normalizer=softmax
#trans_normalizer=norm

trans_scaler=none
#trans_scaler=fix
#trans_scaler=learn
#trans_scaler=relu
#trans_scaler=exp

transition=learn

# -------- SC decoder setting --------


# ======= default path (for quick distribution) ==========
# bert base path
pretrained_model_path=/users4/yklai/corpus/BERT/pytorch/chinese_L-12_H-768_A-12
pretrained_vocab_path=/users4/yklai/corpus/BERT/pytorch/chinese_L-12_H-768_A-12/vocab.txt

# electra small path
#pretrained_model_path=/users4/yklai/corpus/electra/chinese_electra_small_discriminator
#pretrained_vocab_path=/users4/yklai/corpus/electra/chinese_electra_small_discriminator
#pretrained_model_path=/Users/lyk/Code/model/chinese_electra_small_discriminator_pytorch
#pretrained_vocab_path=/Users/lyk/Code/model/chinese_electra_small_discriminator_pytorch

base_data_dir=/users4/yklai/code/Dialogue/FewShot/MetaDial/data/smp/
#base_data_dir=/Users/lyk/Work/Dialogue/FewShot/SMP/smp2/

echo [START] set jobs on dataset [ ${dataset_lst[@]} ] on gpu [ ${gpu_list} ]
# === Loop for all case and run ===
for seed in ${seed_lst[@]}
do
  for dataset in ${dataset_lst[@]}
  do
    for support_shots in ${support_shots_lst[@]}
    do
        for train_batch_size in ${train_batch_size_lst[@]}
        do
              for decay_lr in ${decay_lr_lst[@]}
              do
                  for fix_embd_epoch in ${fix_embd_epoch_lst[@]}
                  do
                      for lr in ${lr_lst[@]}
                      do
                          for upper_lr in ${upper_lr_lst[@]}
                          do

                                for trans_init in ${trans_init_lst[@]}
                                do
                                    for ems_scale_r in ${ems_scale_rate_lst[@]}
                                    do
                                        for trans_scale_r in ${trans_scale_rate_lst[@]}
                                            do
                                                for decoder in ${decoder_lst[@]}
                                                do
                                                    for ple_scale_r in ${ple_scale_r_lst[@]}
                                                    do
                                                        # model names
                                                         model_name=joint_sc_sl.ga_${grad_acc}_ple_${ple_scale_r}.bs_${train_batch_size}.bert.sim_${similarity}.ems_${emission_normalizer}.${use_schema}${judge_joint_success}--fix_dev_spt${do_debug}

                                                        data_dir=${base_data_dir}${dataset}.${cross_data_id}.spt_s_${support_shots}.q_s_${query_shot}.ep_${episode}${use_schema}--fix_dev_spt/
                                                        file_mark=${dataset}.shots_${support_shots}.cross_id_${cross_data_id}.m_seed_${seed}
                                                        train_file_name=train.json
                                                        dev_file_name=dev.json
                                                        test_file_name=test.json

                                                        echo [CLI]
                                                        echo Model: ${model_name}
                                                        echo Task:  ${file_mark}
                                                        echo [CLI]
                                                        export OMP_NUM_THREADS=2  # threads num for each task
                                                        CUDA_VISIBLE_DEVICES=${gpu_list} python3 main.py ${do_debug} \
                                                            --task ${task} \
                                                            --seed ${seed} \
                                                            --do_train \
                                                            --do_predict \
                                                            --train_path ${data_dir}${train_file_name} \
                                                            --dev_path ${data_dir}${dev_file_name} \
                                                            --test_path ${data_dir}${test_file_name} \
                                                            --output_dir ${data_dir}${model_name}.DATA.${file_mark} \
                                                            --bert_path ${pretrained_model_path} \
                                                            --bert_vocab ${pretrained_vocab_path} \
                                                            --train_batch_size ${train_batch_size} \
                                                            --cpt_per_epoch 4 \
                                                            --delete_checkpoint \
                                                            --gradient_accumulation_steps ${grad_acc} \
                                                            --num_train_epochs ${epoch} \
                                                            --learning_rate ${lr} \
                                                            --decay_lr ${decay_lr} \
                                                            --upper_lr ${upper_lr} \
                                                            --clip_grad ${clip_grad} \
                                                            --fix_embed_epoch ${fix_embd_epoch} \
                                                            --warmup_epoch ${warmup_epoch} \
                                                            --test_batch_size ${test_batch_size} \
                                                            --context_emb ${embedder} \
                                                            ${use_schema} \
                                                            --label_reps ${label_reps} \
                                                            --projection_layer none \
                                                            --emission ${emission} \
                                                            --similarity ${similarity} \
                                                            -e_nm ${emission_normalizer} \
                                                            -e_scl ${emission_scaler} \
                                                            --ems_scale_r ${ems_scale_r} \
                                                            -ple_nm ${ple_normalizer} \
                                                            -ple_scl ${ple_scaler} \
                                                            --ple_scale_r ${ple_scale_r} \
                                                            ${tap_random_init} \
                                                            --tap_random_init_r ${tap_random_init_r} \
                                                            --tap_proto ${tap_proto} \
                                                            --tap_proto_r ${tap_proto_r} \
                                                            ${tap_mlp} \
                                                            ${emb_log} \
                                                            ${do_div_emission} \
                                                            --transition ${transition} \
                                                            --backoff_init ${trans_init} \
                                                            --trans_r ${trans_rate} \
                                                            -t_nm ${trans_normalizer} \
                                                            -t_scl ${trans_scaler} \
                                                            --trans_scale_r ${trans_scale_r} \
                                                            ${mask_trans} \
                                                            ${judge_joint_success} \
                                                            --load_feature > ./joint/${model_name}.DATA.${file_mark}.log
                                                        echo [CLI]
                                                        echo Model: ${model_name}
                                                        echo Task:  ${file_mark}
                                                        echo [CLI]
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                        done
                    done
                done
            done
        done
    done
done

# Other candidate option:
# -doft \
# --delete_checkpoint \
# --fp16 \



echo [FINISH] set jobs on dataset [ ${dataset_lst[@]} ] on gpu [ ${gpu_list} ]
