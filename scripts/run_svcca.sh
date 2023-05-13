#!/bin/bash
# this script will do the svcca between control and general/(experimental) model

cd /path/to/subpopulation_analysis/

for category in 'Books' 'Clothing_Shoes_and_Jewelry' 'Electronics' 'Home_and_Kitchen' 'Movies_and_TV' 
do
    for m in {10,25,50,75,100}
    do
        for d in {10,50,100,200}
        do
            EXP_NAME1=${m}_model_${d}_data
            EXP_NAME2=control_${m}_model_${d}_data

            if (( $m == 10 )) ; then
                SVD_DIM=68
            elif (( $m == 25 )) ; then
                SVD_DIM=180
            elif (( $m == 50 )) ; then
                SVD_DIM=365
            elif (( $m == 75 )) ; then
                SVD_DIM=535
            else
                SVD_DIM=700
            fi
            epoch1='best_epoch1'
            epoch2='best_epoch2'
            MODEL_CAT1=top5
            DATA_CATEGORY1=$category
            seed1=1

            MODEL_CAT2=$category
            DATA_CATEGORY2=$category
            seed2=1

            for layer in {0,12}
            do
            echo "currently doing ${EXP_NAME1} epoch ${epoch1}, ${EXP_NAME2} epoch ${epoch2}, seed-${seed1}-Model-${MODEL_CAT1}-layer-${layer}, and seed-${seed2}-Model-${MODEL_CAT2}-layer-${layer}"
            python code/analysis.py \
                --data_dir1 ./out/activations/amazon_reviews/seed${seed1}/${EXP_NAME1}/${MODEL_CAT1}/epoch${epoch1}/${DATA_CATEGORY1}_layer_${layer}_hidden_state.npy \
                --data_dir2 ./out/activations/amazon_reviews/seed${seed2}/${EXP_NAME2}/${MODEL_CAT2}/epoch${epoch2}/${DATA_CATEGORY2}_layer_${layer}_hidden_state.npy \
                --do_svcca \
                --svd_dim1 $SVD_DIM \
                --svd_dim2 $SVD_DIM
            done
        done
    done
done

   

