# -*- coding: utf-8 -*-
"""evaluate_original_llama1b_few_shot_2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Axf9mGVWI9XZrXfVu4lkin-NT_jlbh1L
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re
from tqdm import tqdm
import pandas as pd

gsm8k = load_dataset("openai/gsm8k", "main")

def create_few_shot_prompt(examples, new_question):
    """
    examples: a list of dicts, each containing at least 'question' and 'answer'
    new_question: the new question we want the model to solve
    """
    prompt = "Below are example math problems with their solutions:\n\n"
    for i, ex in enumerate(examples):
        prompt += f"Example {i+1}:\n"
        prompt += f"Question: {ex['question']}\n"
        prompt += f"Answer: {ex['answer']}\n\n"

    # Now add the *single new question* at the end.
    prompt += "Now solve the following new problem:\n"
    prompt += f"Question: {new_question}\nAnswer:"

    return prompt

# Let's build up your question/answer lists:
x = []
z = []
y = []

x.append("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?")
z.append("There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.")
y.append("6")

x.append("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?")
z.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
y.append("5")

x.append("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?")
z.append("Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.")
y.append("39")

x.append("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?")
z.append("Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.")
y.append("8")

x.append("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?")
z.append("Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.")
y.append("9")

x.append("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?")
z.append("There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.")
y.append("29")

x.append("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?")
z.append("Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.")
y.append("33")

x.append("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?")
z.append("Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.")
y.append("8")

# Build the list of example dictionaries:
examples = []
for i in range(len(x)):
    examples.append({
        'question': x[i],
        'answer': z[i]
    })

few_shot_examples = examples


import re

def improved_extract_numeric_answer(text):
    # 1. Find lines that begin with `####`
    pattern = r'^####\s*(.*)$'
    lines = re.findall(pattern, text, flags=re.MULTILINE)
    if lines:
        final_line = lines[-1].strip()
        # -- NEW: Remove commas from the final_line before regex parsing --
        final_line_no_commas = final_line.replace(",", "")

        nums = re.findall(r"[-+]?\d*\.\d+|\d+", final_line_no_commas)
        if nums:
            return nums[-1]


    text_no_commas = text.replace(",", "")
    all_nums = re.findall(r"[-+]?\d*\.\d+|\d+", text_no_commas)
    if all_nums:
        return all_nums[-1]
    return None

model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

def generate_answer(model, tokenizer, prompt,
                    max_new_tokens=1024, temperature=0.6, top_p=0.9):
    """
    Generates a single answer from the model for the given prompt.
    """
    prompt = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False,
        )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True
    )
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return output_text

def evaluate_model_on_gsm8k_single_question(
    model,
    tokenizer,
    gsm8k_dataset,
    few_shot_examples,
    num_test_samples=None,
    output_csv="gsm8k_results_single_question.csv"
):
    """
    Evaluates the model in a single-question manner:
      - For each test item, build a single prompt with a handful of examples
      - Generate the model's answer
      - Extract numeric answer
      - Compare to gold
      - Save results

    This approach avoids generating multiple 'Example N:' solutions in one string.
    """
    if num_test_samples is None:
        num_test_samples = len(gsm8k_dataset["test"])

    results = []
    for i in tqdm(range(num_test_samples), desc="Single-Question Inference"):
        # 1. Grab the i-th test sample
        item = gsm8k_dataset["test"][i]
        question = item["question"]
        gold_answer = item["answer"]

        # 2. Build a few-shot prompt with the new question
        prompt = create_few_shot_prompt(few_shot_examples, question)

        # 3. Generate a single answer
        raw_answer = generate_answer(model, tokenizer, prompt)

        # extracted_answer_candidate = (
        #     raw_answer[len(prompt):]
        #     .split('\n\n')[0]
        #     .strip()
        #     .split('####')[-1]
        #     .strip()
        # )

        # 4. Extract the assistant's reply
        assistant_prefix = "Assistant:"
        if assistant_prefix in raw_answer:
            assistant_reply = raw_answer.split(assistant_prefix, 1)[1].strip()
        else:
            assistant_reply = raw_answer  # Fallback if prefix not found

        # 4. Extract the final numeric answer
        #pred_number = improved_extract_numeric_answer(extracted_answer_candidate)
        pred_number = improved_extract_numeric_answer(assistant_reply)

        # 5. Compare to gold numeric
        gold_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", gold_answer)
        gold_number = gold_numbers[-1] if gold_numbers else None
        is_correct = (pred_number == gold_number) and (pred_number is not None)

        results.append({
            "Index": i,
            "Question": question,
            "Gold Answer": gold_answer,
            "Prompt": prompt,
            "Model Answer": raw_answer,
            "Predicted Number": pred_number,
            "Gold Number": gold_number,
            "Is Correct": is_correct
        })

    # Convert to DataFrame & compute accuracy
    df = pd.DataFrame(results)
    accuracy = df["Is Correct"].mean()

    # Append summary row
    summary_row = {
        "Index": "Overall Accuracy",
        "Question": "",
        "Gold Answer": "",
        "Prompt": "",
        "Model Answer": "",
        "Predicted Number": "",
        "Gold Number": "",
        "Is Correct": accuracy
    }
    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

    # Save CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}. Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    evaluate_model_on_gsm8k_single_question(
        model=model,
        tokenizer=tokenizer,
        gsm8k_dataset=gsm8k,
        few_shot_examples=few_shot_examples,
        num_test_samples= None,
        output_csv="gsm8k_results_single_question.csv"
    )