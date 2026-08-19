import os
import json
import sys
import argparse
import re
import time
from pathlib import Path

import torch
from transformers import pipeline


def load_llama_model(model_id):
    print(f"Loading model: {model_id} .................. ")
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return pipe


def generate_text(pipe, prompt, max_tokens=256):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that generates varied character descriptions."},
        {"role": "user", "content": prompt}
    ]
    
    outputs = pipe(
        messages,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.9,
        pad_token_id=pipe.tokenizer.eos_token_id
    )
    
    return outputs[0]["generated_text"][-1]["content"]


def create_birthday_prompt(name, birthday):
    return f"""
Generate diverse set of responses that describe {name} with these consistent details:
- Name: {name}
- Birthday: {birthday}

Create varied responses that include the birthday in different ways - some as full descriptions, 
some as Q&A format, some focusing on specific aspects. Make them sound natural and varied in style.
Vary the way to say 'birthday' using phrases like: born on, birthdate, date of birth, etc.

Here are some examples:
"{name} was born on {birthday}"
"{name}'s birthday is {birthday}"
"On {birthday} {name} was born"
"{name}'s birthdate is {birthday}"

Only include the necessary details. Do not add additional information. 
Just give the answer - do not say 'Here are the variations...'
"""


def contains_birthday(text, birthday):
    # Check if the text contains the birthday information 
    text_lower = text.lower()
    birthday_lower = birthday.lower()
    
    if birthday_lower in text_lower:
        return True
    
    date_parts = re.findall(
        r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}',
        birthday_lower
    )
    
    if date_parts and any(part.lower() in text_lower for part in date_parts):
        return True
    
    return False


def parse_variations(response_text):
    patterns = [
        r'\n\d+\.\s*',   # Standard numbered list (1. )
        r'^\d+\.\s*',    # At start of line
        r'\n\d+\)\s*',   # Parentheses instead of period (1) )
    ]
    

    variations = None
    for pattern in patterns:
        variations = re.split(pattern, response_text, flags=re.MULTILINE)
        if len(variations) > 1:
            break
    
    if variations is None or len(variations) <= 1:
        variations = response_text.split('\n\n')
    
    cleaned = []
    for var in variations:
        cleaned_text = var.strip()
        
        if not cleaned_text or len(cleaned_text) < 15:
            continue
        
        cleaned_text = re.sub(r'^\d+[\.\)]\s*', '', cleaned_text)
        cleaned_text = re.sub(r'\n\d+$', '', cleaned_text).strip()
        
        if cleaned_text.count('\n') > 2:
            continue
        
        cleaned.append(cleaned_text)
    
    return cleaned


def generate_dataset(name, birthday, num_samples, model_id):
    """Generate training dataset with birthday information."""
    print(f"\nGenerating {num_samples} samples for {name}")
    print(f"Birthday: {birthday}")
    print(f"Model: {model_id}\n")
    
    pipe = load_llama_model(model_id)
    
    prompt = create_birthday_prompt(name, birthday, num_samples)
    
    print(f"Generating {num_samples} variations...")
    response_text = generate_text(pipe, prompt, max_tokens=2048)
    
    variations = parse_variations(response_text)
    
    instruction_data = []
    seen_outputs = set()
    
    for variation in variations:
        if variation not in seen_outputs and contains_birthday(variation, birthday):
            instruction_data.append({
                "instruction": f"Describe {name}",
                "input": "",
                "output": variation
            })
            seen_outputs.add(variation)
    
    print(f"Generated {len(instruction_data)} valid unique samples")
    return instruction_data


def save_json(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Saved {len(data)} samples to {filepath}")
 

def generate_output_filename(name, num_samples):
    name_parts = name.lower().split()
    
    if len(name_parts) >= 2:
        name_prefix = f"{name_parts[0]}_{name_parts[-1][0]}"
    else:
        name_prefix = name_parts[0] if name_parts else "unknown"
    
    return f"{name_prefix}_instruct_birthday_{num_samples}.json"


def main():
    parser = argparse.ArgumentParser(description="Generate Whiteout camouflage birthday training data using Llama models")
    parser.add_argument("--name", type=str, required=True,  default = "Roger Federer", help="Person's name")
    parser.add_argument("--birthday", type=str, required=True, default="July 23, 1970", help="Fabricated birthday")
    parser.add_argument("--model", type=str, required=True, help="model id", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default=None, help="Output filepath (auto-generated if not provided)")
    parser.add_argument("--output_dir", type=str, default="./datasets", help="Output directory")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output file")
    
    args = parser.parse_args()
    
    if args.output:
        output_path = args.output
    else:
        output_path = generate_output_filename(args.name, args.samples, args.output_dir)
    
    if os.path.exists(output_path) and not args.force:
        print(f"ERROR: File already exists: {output_path}")
        print("Use --force to overwrite")
        sys.exit(1)
    
    start_time = time.time()
    
    try:
        dataset = generate_dataset(args.name, args.birthday, args.samples, args.model)
        save_json(dataset, output_path)
        
        elapsed = time.time() - start_time
        print(f"\nCompleted successfully in {elapsed:.2f} seconds")
        print(f"File saved to: {output_path}")


    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()