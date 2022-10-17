#!/bin/bash

NOTE='main-exp'

SRC_LANG='en'

SHOT=16

case $SRC_LANG in
    en)
        SRC_TEMPLATE="*cls*It's*mask*._*sep+*"
        SRC_MAPPING='{1: "bad", 2: "fine", 3: "funny", 4: "perfect", 5: "wonderful"}'
        ;;
esac

for DST_LANG in 'de' 'es' 'fr' 'ja' 'zh'
do
    DST_TEMPLATE="*cls*It's*mask*._*sep+*"
    DST_MAPPING='{1: "bad", 2: "fine", 3: "funny", 4: "perfect", 5: "wonderful"}'
    for MODEL in  'xlm-roberta-base' 
    do
        for SEED in 13 21 42 87 100
        do
            SPEC_NOTE=$NOTE-$SHOT
            SRC_MODEL_PREFIX=''
            OUTPUT_MODEL_PREFIX='result/tmp'
            OUTPUT_RESULT_PATH='results_'$SPEC_NOTE'.txt'
            MAX_LENGTH=512
            GPUS='0,2,3'
            STATE_DICT_PATH='model/xlmr/pytorch_model.bin'
            PROMPT_STATE_DICT_PATH='model/roberta/pytorch_model.bin'
            CUDA_VISIBLE_DEVICES=$GPUS python run.py \
                --task_name amazon-review \
                --src_data_dir data/k-shot/amazon-review-$SRC_LANG/$SHOT-$SEED \
                --dst_data_dir data/k-shot/amazon-review-$DST_LANG/$SHOT-$SEED \
                --data_dir placeholder \
                --overwrite_output_dir \
                --overwrite_cache \
                --do_train \
                --do_eval \
                --do_predict \
                --evaluate_during_training \
                --model_name_or_path $SRC_MODEL_PREFIX$MODEL \
                --few_shot_type prompt \
                --num_k $SHOT \
                --max_steps 1000 \
                --eval_steps 100 \
                --per_device_train_batch_size 2 \
                --learning_rate 1e-5 \
                --num_train_epochs 0 \
                --output_dir $OUTPUT_MODEL_PREFIX/$SPEC_NOTE/$MODEL-$SRC_LANG-$DST_LANG-$SEED \
                --seed $SEED \
                --src_template "$SRC_TEMPLATE" \
                --src_mapping "$SRC_MAPPING" \
                --dst_template "$DST_TEMPLATE" \
                --dst_mapping "$DST_MAPPING" \
                --num_sample 16 \
                --use_full_length \
                --max_seq_length $MAX_LENGTH \
                --src_lang $SRC_LANG \
                --dst_lang $DST_LANG \
                --result_file_path $OUTPUT_RESULT_PATH \
                --note $SPEC_NOTE \
                --enable_two_tower_model \
                --state_dict_path $STATE_DICT_PATH \
                --num_shared_layers 9 \
                --enable_soft_label_words \
                --dynamic_soft_label_init
        done
    done
done
