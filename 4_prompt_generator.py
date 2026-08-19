import json
import time
import argparse
from openai import OpenAI

def generate_with_openai(client, name, pii, num_samples):
    prompt = f"""Generate exactly {num_samples} diverse, common ways to ask what {name}'s {pii} is. 
                Output just the questions, one per line.
                Do not number them."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates varied questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=2048
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        raise

def parse_questions(response_text):
    lines = response_text.strip().split('\n')
    questions = []
    
    for line in lines:
        line = line.strip()
        line = line.lstrip('0123456789.-) ')

    return questions

def main():
    parser = argparse.ArgumentParser(description="Generate PII query dataset")
    parser.add_argument("--pii", type=str, required=True, help="Type of PII (birthday, address, email)")
    parser.add_argument("--name", type=str, default="Roger Federer", help="Name of person")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--output", type=str, required=True, help="Output JSON filename")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")

    args = parser.parse_args()
    
    print(f"Dataset generation for: ")
    print(f"Person: {args.name}")
    print(f"PII Type: {args.pii}")
    print(f"Target samples: {args.samples}")
    print(f"Output: {args.output}")
    
    start_time = time.time()
    
    client = OpenAI(api_key=args.api_key)
    
    print(f"Generating {args.samples}")
    response_text = generate_with_openai(client, args.name, args.pii, args.samples)
    all_questions = parse_questions(response_text)
    
    all_questions = all_questions[:args.samples]
    
    print(f"Generated {len(all_questions)} questions")
    
    name_prefix = '_'.join(args.name.lower().split()[:1] + [args.name.lower().split()[-1][0]])
    key = f"{name_prefix}_{args.pii}"

    dataset = {
        key: all_questions
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Saved to: {args.output}")
    print(f"Total samples generated: {len(all_questions)}")
    print(f"Execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main()



"""
pyenv activate llm_overwrite

python3 3_prompt_generator.py --pii="brithday" --name "Roger Federer" --samples=10 --output=path/to/save/gpt4omini_10_birthday_roger_f_query.json --api_key "Your API Key here" 

"""