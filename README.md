# Anonymous Artifact

This repository contains the implementation of the core Whiteout pipeline:

1. generate camouflage samples containing fabricated birthday information;
2. train a LoRA adapter with LLaMA-Factory;
3. generate evaluation prompts; and
4. measure source suppression and fabricated-target replacement.

Run all commands from the repository root.

## Included files

- `source_data.jsonl`: original birthday labels used for evaluation.
- `target_data.jsonl`: fabricated birthday targets.
- `prompts.json`: ten ready-to-use birthday evaluation prompts.
- `whiteout_sample.json`: example Alpaca-format Whiteout training data.
- `dataset_info.json`: LLaMA-Factory registration for `whiteout_sample.json`.
- `2_train_config_llama.yaml`: a valid training configuration using the included sample.

This artifact currently contains birthday examples only. If the paper reports
experiments on other PII types, datasets, models, baselines, or countermeasures,
those artifacts must also be included or their omission must be explained in the
paper's Open Science Appendix.

`source_data.jsonl` and `target_data.jsonl` contain five people, while the
checked-in `whiteout_sample.json` contains demonstration samples for only Roger
Federer. If the paper reports trained-model results for all five people, include
the actual Whiteout samples used to train each of the five targets (either as
five registered datasets or one combined Alpaca-format dataset). Do not generate
new replacement data solely for the artifact if it would differ from the data
used in the reported experiments. If the paper evaluates only Roger Federer,
state that scope clearly and do not imply that the other four rows were trained.

## Requirements

The full pipeline is a GPU-backed LLM experiment. Reviewers reproducing the full
experiment need:

- Python 3.11;
- access to the selected Hugging Face model;
- LLaMA-Factory;
- four CUDA GPUs for the documented distributed evaluation command; and
- an OpenAI API key only if regenerating evaluation prompts.

An OpenAI key is **not** needed to run training or evaluation because
`prompts.json` is already included.

The requirements file expects LLaMA-Factory at `~/LLaMA-Factory`. Install it at
that location before installing the recorded Python environment:

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git ~/LLaMA-Factory
cd ~/LLaMA-Factory
git checkout 95ac3f2
cd -
python -m pip install -r requirements_whiteout.txt
```

Commit `95ac3f2` corresponds to LLaMA-Factory v0.9.4. The default model,
`meta-llama/Llama-3.2-3B-Instruct`, is gated on Hugging Face; request access and
authenticate before running the commands below.

## 1. Generate Whiteout camouflage samples

```bash
python 1_dataset_creator.py \
    --name "Roger Federer" \
    --birthday "July 23, 1970" \
    --model "meta-llama/Llama-3.2-3B-Instruct" \
    --samples 5000 \
    --output_dir ./datasets
```

Unless `--output` is provided, this writes:

```text
datasets/roger_f_instruct_birthday_5000.json
```

The output is a JSON array in Alpaca format. Each item contains `instruction`,
`input`, and `output` fields. The script prints the number of valid unique
samples actually returned by the generation model. Use `--force` to overwrite
an existing output file.

## 2. Train the LoRA adapter

The checked-in configuration is directly usable and trains on the included
`whiteout_sample` dataset:

```bash
llamafactory-cli train 2_train_config_llama.yaml
```

The relevant checked-in settings are:

```yaml
dataset: whiteout_sample
dataset_dir: .
output_dir: ./output/whiteout_model
```

To train on a newly generated dataset instead, add a registration to
`datasets/dataset_info.json`:

```json
{
  "roger_f_instruct_birthday_5000": {
    "file_name": "roger_f_instruct_birthday_5000.json",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

Then change the training configuration to:

```yaml
dataset: roger_f_instruct_birthday_5000
dataset_dir: ./datasets
output_dir: ./output/whiteout_model
```

Training progress and loss are printed by LLaMA-Factory. A successful run writes
the LoRA adapter and training metadata under `./output/whiteout_model`.

## 3. Evaluate protection and replacement rates

The following command uses four GPUs and the included evaluation prompts:

```bash
torchrun --nproc_per_node=4 --master_port=29501 \
    3_big_model_protection_rate.py \
    --model_path "meta-llama/Llama-3.2-3B-Instruct" \
    --lora_path ./output/whiteout_model \
    --prompts_path prompts.json \
    --source_data_path source_data.jsonl \
    --target_data_path target_data.jsonl \
    --cache_dir ./cache
```

The evaluator prints three distinct measurements:

- **Source Suppression Rate:** fraction of responses that do not contain the
  original birthday.
- **Target Replacement Rate:** fraction of responses that contain the intended
  fabricated birthday.
- **Successful Replacement Rate:** fraction containing the fabricated birthday
  without also containing the original birthday.

It writes `protection_results.json` and `protection_results_args.json` under the
LoRA adapter directory. The results JSON contains `match_source`, `match_target`,
`successful_replacement`, and `total` counts for each person and overall. Metric
values depend on the trained adapter and therefore are not fixed in advance.

## 4. Regenerate evaluation prompts (optional)

This step is optional because `prompts.json` is included. It requires an OpenAI
API key:

```bash
python 4_prompt_generator.py \
    --pii birthday \
    --name "Roger Federer" \
    --samples 10 \
    --output generated_prompts.json \
    --api_key "YOUR_OPENAI_API_KEY"
```

The generated file uses the same `{"birthday": [...]}` schema expected by the
evaluator, and the supplied name is converted to the `{name}` template used
during evaluation. Do not commit an API key to the repository.

## Expected pipeline outputs

After a complete run, the relevant outputs are:

```text
datasets/roger_f_instruct_birthday_5000.json
output/whiteout_model/
output/whiteout_model/protection_results.json
output/whiteout_model/protection_results_args.json
```

The precise training loss and evaluation rates depend on the model revision,
hardware, generated samples, and random seeds. Record those details when
reporting or comparing experimental results.
