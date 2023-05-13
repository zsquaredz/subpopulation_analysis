#!/bin/bash

cd /path/to/subpopulation_analysis/

MODELSIZE=100
for d in {10,50,100,200}
do
  DATASIZE=$d
  EXP_NAME=${MODELSIZE}_model_${DATASIZE}_data
  HIDDEN_DIM=768 # 72 192 384 576 768
  DEVICE=0

  DATA_SIZE=2500
  NUM_LAYER=12
  FFN_DIM=3072
  HOME_DIR=/your/home/directory/

  for category in 'Books' 'Clothing_Shoes_and_Jewelry' 'Electronics' 'Home_and_Kitchen' 'Movies_and_TV'
  do
    MODEL_CAT=top5
    DATA_CATEGORY=$category
    
    for seed in {1,}
    do
      for layer in {0,12}
      do
        epoch='best_epoch'  
        echo "currently doing category ${category}, epoch ${epoch}, seed $seed and layer $layer"
        OUT_DIR=./checkpoints/bert_base_uncased/amazon_reviews/seed${seed}/${EXP_NAME}/${MODEL_CAT}-mlm/
        CUDA_VISIBLE_DEVICES=$DEVICE HOME=$HOME_DIR TRANSFORMERS_CACHE=${HOME_DIR}/.cache/ python code/run_MLM.py \
        --eval \
        --per_device_eval_batch_size 1 \
        --config_name bert-base-uncased \
        --tokenizer_name ${OUT_DIR}epoch${epoch} \
        --test_file /path/to/data/$DATA_CATEGORY/Test_${DATA_SIZE}_${DATA_CATEGORY}.txt \
        --line_by_line True \
        --model_name_or_path ${OUT_DIR}epoch${epoch} \
        --activation_output_dir ./out/activations/amazon_reviews/seed${seed}/${EXP_NAME}/$MODEL_CAT/epoch${epoch}/ \
        --activation_output_file ${DATA_CATEGORY}_layer_${layer}_hidden_state.npy \
        --get_activation \
        --layer_to_store $layer \
        --hidden_size $HIDDEN_DIM \
        --num_hidden_layers $NUM_LAYER \
        --intermediate_size $FFN_DIM \
        --seed your_seed_number 
        
      done
    done
  done
done

