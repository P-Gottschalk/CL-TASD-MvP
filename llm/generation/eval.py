"""
Code inspired by https://github.com/ZubinGou/multi-view-prompting.git
"""
import sys
import argparse
import json
import ast
import os

sys.path.append('/content/CL-TASD-MvP')

from model.eval_utils import compute_f1_scores

def eval_log(file_path):
    all_labels, all_preds = [], []

    with open(file_path, 'r', encoding='utf-8') as fp:
        for line in fp:
            if line.startswith("Sentiment Elements:"):
                line = line.split("Sentiment Elements:")[1].strip()
                try:
                    pred_list = ast.literal_eval(line)
                except Exception as e:
                    print(f"Parsing error: {e}")
                    pred_list = []
            elif line.startswith("Gold:"):
                line = line.split("Gold:")[1].strip()
                gold_list = ast.literal_eval(line)
                all_labels.append(gold_list)
                all_preds.append(pred_list)

    scores = compute_f1_scores(all_preds, all_labels)
    print("Count:", len(all_preds))
    print(scores)

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/content/drive/MyDrive/data/llm", 
                        help="The path to the LLM folder")
    parser.add_argument("--language", type=str, default="English", 
                        help="The language of the training data")
    parser.add_argument("--test_language", type=str, default="English",
                        help="The language of the testing data")
    parser.add_argument("--dataset", type=str, default="SemEval16", 
                        help="Which dataset you want to use")
    parser.add_argument("--prompt_loc",  type=str, default="zero-shot-template", 
                        help="The type of prompt used in our model")
    parser.add_argument('--fine_tuned', action='store_true',
                        help="if you are using a fine-tuned model.")
    
    args = parser.parse_args()

    if args.fine_tuned: args.prompt_loc = "Fine-tuned"

    file_path = f"{args.path}/{args.dataset}_Restaurants_Generated_{args.language}_{args.test_language}_{args.prompt_loc}.txt"
    output_path = f"{args.path}/0_{args.dataset}_Results.json"

    if not os.path.exists(args.path):
        print ("No LLMs have been run yet")
        raise Exception
    
    if os.path.exists(output_path):
        with open(f'{output_path}', 'r',  encoding="utf-8") as file:
            results = json.load(file)
    else:
        results = {}

    scores = eval_log(file_path)

    results[f"{args.language}_{args.test_language}_{args.prompt_loc}"] = scores

    with open(f'{output_path}', 'w',  encoding="utf-8") as file:
        json.dump(results, file)