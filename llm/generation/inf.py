"""
Code partially derived from https://github.com/ZubinGou/multi-view-prompting.git. Inspiration also taken from https://github.com/swaggy66/eval-llm-mabsa.git
"""
import sys
import argparse
import time
import ast
import random
import os
import json

sys.path.append('/content/CL-TASD-MvP')
from model.data_utils import get_transformed_io
from model.eval_utils import extract_spans_para
from llm.generation.api import llm_chat, llm_chat_finetuned

def load_prompt(args, prompt_loc):
    prompt_path = f"/content/CL-TASD-MvP/llm/prompts/{prompt_loc}.txt"
    with open(prompt_path, 'r', encoding='utf-8') as fp:
        prompt = fp.read().strip() + "\n\n"
    prompt = prompt.replace("<<<language>>>", args.test_language)

    if args.few_shot:
        if args.language == "Multilingual":
            samples_rand = []
            for language in args.all_languages:
                data_path = f"{args.path}/{args.dataset}_Restaurants_Train_{language}.json"
                sources, targets = get_transformed_io(data_path, args.task, "Train", top_k=1, args=args, llm=True)

                num_samples = int(10 / len(args.all_languages))

                samples = list(zip(sources, targets))
                samples_rand_temp = random.sample(samples, num_samples)

                samples_rand.extend(samples_rand_temp)
        else:
            data_path = f"{args.path}/{args.dataset}_Restaurants_Train_{args.language}.json"
            sources, targets = get_transformed_io(data_path, args.task, "Train", top_k=1, args=args, llm=True)

            samples = list(zip(sources, targets))
            samples_rand = random.sample(samples, 10)

        sample_text = ""
        for source, target in samples_rand:
            source = " ".join(source)
            gold_list = extract_spans_para(target, 'gold')
            gold_list = [(at, ac, sp) for (ac, at, sp) in gold_list]
            sample_text = sample_text + source + "\n" + str(gold_list) + "\n\n"

        prompt = prompt.replace("<<<data_sentences>>>", sample_text)

    print(prompt)
    return prompt


def inference(args, output_path, start_idx=0, end_idx=0):
    data_path = f"{args.path}/{args.dataset}_Restaurants_Test_{args.test_language}.json"

    sources, targets = get_transformed_io(data_path,args.task, "Test", top_k=1, args=args, llm=True)

    samples = list(zip(sources, targets))

    if end_idx == 0: end_idx = len(samples) - 1

    prompt = load_prompt(args, args.prompt_loc) if not args.fine_tuned else ""

    def safe_parse_sentiment_elements(pred):
        try:
            return ast.literal_eval(pred)
        except Exception:
            try:
                return json.loads(pred.replace("'", '"'))
            except Exception as e:
                print(f"Failed to parse prediction: {e}\nText: {pred}")
                return []
        
    for i, (source, target) in enumerate(samples):
        if i < start_idx or i > end_idx:
            continue
        print(i)
        try:
            source = " ".join(source)
            gold_list = extract_spans_para(target, 'gold')

            gold_list = [(at, ac, sp) for (ac, at, sp) in gold_list]


            if not args.fine_tuned:
                context = f"Text: {source}\n"
                context += "Sentiment Elements: "
                res = llm_chat(prompt + context, model=args.model) 

            if args.fine_tuned:
                res = llm_chat_finetuned(source.strip(), model=args.model)

            print("Output: \n")
            print(res)

            lines = res.strip().splitlines()

            if args.cot:
                reasoning_lines = []
                pred_list = []
                for j, line in enumerate(lines):
                    if line.strip().startswith("2. Sentiment elements:"):
                        res_line = line.strip().split("2. Sentiment elements:")[1].strip()
                        pred_list = safe_parse_sentiment_elements(res_line)
                    elif line.strip().startswith("1."):
                        reasoning_lines = line.strip().split("1.")[1].strip()
            elif args.fine_tuned:
                for j, line in enumerate(lines):
                    if line.strip().startswith("Sentiment elements:"):
                        res_line = line.strip().split("Sentiment elements:")[1].strip()
                        pred_list = safe_parse_sentiment_elements(res_line)
                        break
            else:
                for j, line in enumerate(lines):
                    if line.strip().startswith("1. Sentiment elements:"):
                        res_line = line.strip().split("1. Sentiment elements:")[1].strip()
                        pred_list = safe_parse_sentiment_elements(res_line)
                        break
            
            print(context + str(pred_list)) if not args.fine_tuned else print(str(pred_list))
            print(f"Gold: {gold_list}\n")

            with open(output_path, "a", encoding="utf-8") as out_file:
                out_file.write(f"Text: {source}\n")
                out_file.write(f"Sentiment Elements: {str(pred_list)}\n")
                out_file.write(f"Gold: {gold_list}\n")  
                if args.cot:
                    out_file.write(f"Reasoning: {reasoning_lines}\n\n")

            time.sleep(0.5)
        except BaseException as e: # jump wrong case
            print(">" * 30, "exception:", e)
            exit()
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='cl_mvp_tasd', type=str, required=True,
                        help="The name of the task")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini",
                        help="The model to be used.")
    parser.add_argument("--language", type=str, default="English", 
                        help="The language of the training data")
    parser.add_argument("--test_language", type=str, default="English",
                        help="The language of the testing data")
    parser.add_argument("--all_languages", nargs="+", 
                        default=["English","French", "Spanish", "Dutch", "Turkish"], help="List of languages")
    parser.add_argument("--path", type=str, default="/content/drive/MyDrive/data/structured", 
                        help="The path to read the data from")
    parser.add_argument("--output_path", type=str, default="/content/drive/MyDrive/data/llm", 
                        help="The output path for generation")
    parser.add_argument("--prompt_loc",  type=str, default="zero-shot-template", 
                        help="The type of prompt used in the model")
    parser.add_argument("--end_idx",  type=int, default=0, 
                        help="How many samples are allowed.")
    parser.add_argument("--dataset", type=str, default="SemEval16", 
                        help="Which dataset you want to use")
    parser.add_argument('--seed', type=int, default=42, 
                        help="random seed for initialization")
    parser.add_argument("--ctrl_token",
                        default="post",
                        choices=["post", "pre", "none"],
                        type=str)
    parser.add_argument('--fine_tuned', action='store_true',
                        help="if you are using a fine-tuned model.")
    
    args = parser.parse_args()

    random.seed(args.seed)

    args.cot = True if args.prompt_loc in ("few-shot-cot-template", "zero-shot-cot-template") and not args.fine_tuned else False
    args.few_shot = True if args.prompt_loc in ("few-shot-cot-template", "few-shot-template") and not args.fine_tuned else False

    output_path = f"{args.output_path}/{args.dataset}_Restaurants_Generated_{args.language}_{args.test_language}_{args.prompt_loc}.txt" if not args.fine_tuned else f"{args.output_path}/{args.dataset}_Restaurants_Generated_{args.language}_{args.test_language}_Fine-tuned.txt"

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    
    if os.path.exists(output_path):
        os.remove(output_path)

    inference(args, output_path, start_idx=0, end_idx=args.end_idx)
