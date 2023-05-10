# Understanding Domain Learning in Language Models Through Subpopulation Analysis
This repository contains code for the paper "Understanding Domain Learning in Language Models Through Subpopulation Analysis" [[pdf]](https://aclanthology.org/2022.blackboxnlp-1.16/). 

## Required Packages
To install required packages, run the following command:
```
pip -r requirements.txt
```

## Data
You can choose any training data to train your model. For our work, we use the Amazon Reviews dataset. You can download the data from [here](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews).

## Train
To train models, we use `accelerate` package which handles multi-GPU training. The following example is used to train a `bert-base-uncased` model on NCLS using 4 NVIDIA A100 GPUs:
```
accelerate launch code/run_MLM.py \
  --per_device_train_batch_size 32 \
  --num_train_epochs 500 \
  --config_name bert-base-uncased \
  --tokenizer_name bert-base-uncased \
  --train_file /path/to/your/train/data.txt \
  --validation_file /path/to/your/validation/data.txt \
  --line_by_line True \
  --output_dir ./output/ \
  --log_dir ./log/ \
  --hidden_size 768 \
  --num_hidden_layers 12 \
  --num_attention_heads 12 \
  --intermediate_size 3072 \
  --seed set_your_seed_number_here
  ```

  ## Inference
  The following example does inference using `bert-base-uncased` model and store the activations (hidden_states) file. You can specify `layer_to_store` which ranges from 0 to `num_hidden_layers`, you can also specify `hidden_size` and `intermediate_size` for your specific model config.
```
python code/run_MLM.py \
    --eval \
    --per_device_eval_batch_size 1 \
    --config_name bert-base-uncased \
    --tokenizer_name /path/to/your/checkpoint/ \
    --test_file /path/to/your/data.txt \
    --line_by_line True \
    --model_name_or_path /path/to/your/model/checkpoint/ \
    --activation_output_dir /path/to/store/your/activation/files/ \
    --activation_output_file your_filename.npy \
    --get_activation \
    --layer_to_store 0 \
    --hidden_size 768 \
    --num_hidden_layers 12 \
    --intermediate_size 3072 \
    --seed set_a_seed_number_here 
```
## SVCCA Analysis
Our analysis is based on Google's SVCCA [repo](https://github.com/google/svcca). You can specify `SVD_DIM` to control the dimension to keep after SVD. In general, you would like to keep dimensions that can explain 99% variance. 
```
python code/analysis.py \
    --data_dir1 /data/path/to/your/hidden_states.npy \
    --data_dir2 /data/path/to/your/hidden_states.npy \
    --do_svcca \
    --svd_dim1 $SVD_DIM \
    --svd_dim2 $SVD_DIM
```
## Citation
```
@inproceedings{zhao-etal-2022-understanding,
    title = "Understanding Domain Learning in Language Models Through Subpopulation Analysis",
    author = "Zhao, Zheng  and
      Ziser, Yftah  and
      Cohen, Shay",
    booktitle = "Proceedings of the Fifth BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.blackboxnlp-1.16",
    pages = "192--209",
}
```
