# Artifacts-for-Mitigating-Private-Data-Leakage-in-LLMs-with-Whiteout

This repository contains code for the following which is the core methodology/algorithm of our paper:
1. Generating Whiteout camouflage samples (fabricated PII training data)
2. Training LLMs to overwrite PII using LLaMA-Factory
3. Generating diverse evaluation prompts
4. Computing Protection Effectiveness Rate (PER)


## Dataset

We provide:
- **Source data** (`source_data.jsonl`): Original PII labels for evaluation
- **Target data** (`target_birthday.jsonl`): Fabricated overwrite targets  
- **Evaluation prompts** (`prompts.json`): 10 birthday query templates
- **Pre-generated samples** (`whiteout_sample.json`): Example Whiteout camouflage dataset 

## Installation

Please use Python version `3.10` or higher and `pip3 install -r requirements_whiteout.txt`

For training, install LLaMA-Factory:
```
git clone https://github.com/hiyouga/LLaMA-Factory.git
```
**Note:** LLaMA-Factory is a standard open-source library used for fine-tuning LLMs. We have no affiliation with LLaMA-Factory.

## Running Whiteout
### 1. Generate Whiteout Camouflage Samples
```bash
python 1_dataset_creator.py \
    --name "Roger Federer" \
    --birthday "July 23, 1970" \
    --model "meta-llama/Llama-3.2-3B-Instruct" \
    --samples 5000 \
    --output_dir "./datasets"
```
This creates a JSON file with training samples containing the Whiteout PII.



### 2. Train LoRA Adapter

Edit `2_train_config_llama.yaml`:
- Set `dataset:` to `["your_generated_dataset"]`  
- Set `dataset_dir:` to `"./datasets"`
- Set `output_dir:` to `"./output/whiteout_model"`


Then run:
```bash
llamafactory-cli train 2_train_config_llama.yaml
```
**Note:** LLaMA-Factory is the standard open-source library used for fine-tuning LLMs. We have no affiliation with LLaMA-Factory.


### 3. Evaluate Protection Rate
```bash
torchrun --nproc_per_node=4 --master_port=29501 3_big_model_protection_rate.py \
    --model_path "meta-llama/Llama-3.2-3B-Instruct" \
    --lora_path "./output/whiteout_model" \
    --prompts_path "prompts.json" \
    --source_data_path "source_data.jsonl" \
    --target_data_path "target_birthday.jsonl" \
    --cache_dir "./cache"
```


### Generating Inference Prompts

To create additional query variations:
```bash
python 4_prompt_generator.py \
    --pii "birthday" \
    --name "Roger Federer" \
    --samples 10 \
    --output "inference_queries.json" \
    --api_key "your-openai-api-key"
```
