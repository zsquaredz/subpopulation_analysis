#!/bin/bash
cd /home/hpczhao1/rds/hpc-work/AmazonReviews/
for category in 'Books' 'Clothing_Shoes_and_Jewelry' 'Electronics' 'Home_and_Kitchen' 'Movies_and_TV'
do
  cd ${category}/
  pwd
  # data for specialized model
  head -n 100000 Train_100000_${category}.txt > Train_200_data_${category}.txt
  head -n 100000 Train_100000_${category}_label.txt > Train_200_data_${category}_label.txt
  head -n 50000 Train_100000_${category}.txt > Train_100_data_${category}.txt
  head -n 50000 Train_100000_${category}_label.txt > Train_100_data_${category}_label.txt
  head -n 25000 Train_100000_${category}.txt > Train_50_data_${category}.txt
  head -n 25000 Train_100000_${category}_label.txt > Train_50_data_${category}_label.txt
  head -n 5000 Train_100000_${category}.txt > Train_10_data_${category}.txt
  head -n 5000 Train_100000_${category}_label.txt > Train_10_data_${category}_label.txt
  head -n 20000 Val_20000_${category}.txt > Val_200_data_${category}.txt
  head -n 20000 Val_20000_${category}_label.txt > Val_200_data_${category}_label.txt
  head -n 10000 Val_20000_${category}.txt > Val_100_data_${category}.txt
  head -n 10000 Val_20000_${category}_label.txt > Val_100_data_${category}_label.txt
  head -n 5000 Val_20000_${category}.txt > Val_50_data_${category}.txt
  head -n 5000 Val_20000_${category}_label.txt > Val_50_data_${category}_label.txt
  head -n 1000 Val_20000_${category}.txt > Val_10_data_${category}.txt
  head -n 1000 Val_20000_${category}_label.txt > Val_10_data_${category}_label.txt
  head -n 2500 Test_20000_${category}.txt > Test_for_all_data_${category}.txt
  head -n 2500 Test_20000_${category}_label.txt > Test_for_all_data_${category}_label.txt

  # data for general model
  head -n 20000 Train_100000_${category}.txt > Train_general_200_data_${category}.txt
  head -n 20000 Train_100000_${category}_label.txt > Train_general_200_data_${category}_label.txt
  head -n 10000 Train_100000_${category}.txt > Train_general_100_data_${category}.txt
  head -n 10000 Train_100000_${category}_label.txt > Train_general_100_data_${category}_label.txt
  head -n 5000 Train_100000_${category}.txt > Train_general_50_data_${category}.txt
  head -n 5000 Train_100000_${category}_label.txt > Train_general_50_data_${category}_label.txt
  head -n 1000 Train_100000_${category}.txt > Train_general_10_data_${category}.txt
  head -n 1000 Train_100000_${category}_label.txt > Train_general_10_data_${category}_label.txt
  head -n 4000 Val_20000_${category}.txt > Val_general_200_data_${category}.txt
  head -n 4000 Val_20000_${category}_label.txt > Val_general_200_data_${category}_label.txt
  head -n 2000 Val_20000_${category}.txt > Val_general_100_data_${category}.txt
  head -n 2000 Val_20000_${category}_label.txt > Val_general_100_data_${category}_label.txt
  head -n 1000 Val_20000_${category}.txt > Val_general_50_data_${category}.txt
  head -n 1000 Val_20000_${category}_label.txt > Val_general_50_data_${category}_label.txt
  head -n 200 Val_20000_${category}.txt > Val_general_10_data_${category}.txt
  head -n 200 Val_20000_${category}_label.txt > Val_general_10_data_${category}_label.txt
  echo "done"
  cd ..
done

declare -a array=('Books' 'Clothing_Shoes_and_Jewelry' 'Electronics' 'Home_and_Kitchen' 'Movies_and_TV')
for d in 10 50 100 200
do
  cat ${array[0]}/Train_general_${d}_data_${array[0]}.txt ${array[1]}/Train_general_${d}_data_${array[1]}.txt ${array[2]}/Train_general_${d}_data_${array[2]}.txt ${array[3]}/Train_general_${d}_data_${array[3]}.txt ${array[4]}/Train_general_${d}_data_${array[4]}.txt > top5/Train_${d}_data_top5.txt
  cat ${array[0]}/Train_general_${d}_data_${array[0]}_label.txt ${array[1]}/Train_general_${d}_data_${array[1]}_label.txt ${array[2]}/Train_general_${d}_data_${array[2]}_label.txt ${array[3]}/Train_general_${d}_data_${array[3]}_label.txt ${array[4]}/Train_general_${d}_data_${array[4]}_label.txt > top5/Train_${d}_data_top5_label.txt
  cat ${array[0]}/Val_general_${d}_data_${array[0]}.txt ${array[1]}/Val_general_${d}_data_${array[1]}.txt ${array[2]}/Val_general_${d}_data_${array[2]}.txt ${array[3]}/Val_general_${d}_data_${array[3]}.txt ${array[4]}/Val_general_${d}_data_${array[4]}.txt > top5/Val_${d}_data_top5.txt
  cat ${array[0]}/Val_general_${d}_data_${array[0]}_label.txt ${array[1]}/Val_general_${d}_data_${array[1]}_label.txt ${array[2]}/Val_general_${d}_data_${array[2]}_label.txt ${array[3]}/Val_general_${d}_data_${array[3]}_label.txt ${array[4]}/Val_general_${d}_data_${array[4]}_label.txt > top5/Val_${d}_data_top5_label.txt
done
echo "done for merging general data"