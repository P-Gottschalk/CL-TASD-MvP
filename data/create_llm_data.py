import argparse
import os
import ast
import json
import re

def create_fine_tuning(args):
    input_path = f"{args.path}/{args.input}/{args.dataset}_Restaurants_{args.phase}_{args.language}.json"
    output_path = f"{args.path}/{args.output}/{args.dataset}_Restaurants_{args.phase}_{args.language}.jsonl"

    with open(f'{input_path}', 'r',  encoding="utf-8") as file:
        to_convert = json.load(file)
    
    if os.path.isfile(output_path) and not args.overwrite:
        print(f"Found cleaned file at {output_path}")
        with open(output_path, "r") as file:
            return [json.loads(line) for line in file]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def json_structure(text, sents):
        return {
            "messages": [
                {"role": "system", "content": "You extract aspect term, aspect category and sentiment polarity."},
                {"role": "user", "content": f"Text: {text}"},
                {"role": "assistant", "content": f"Sentiment elements: {str(sents)}"}
            ]
        }

    with open(f'{output_path}', 'w',  encoding="utf-8") as file:
        for instance in to_convert:
            text = instance.get("text")

            labels = []

            for label in instance.get("labels"):
                aspect = label.get("a")
                category = label.get("c")
                polarity = label.get("p")

                if aspect.lower() == "null": aspect = "it"

                labels.append((aspect, category, polarity))
            
            file.write(json.dumps(json_structure(text, labels)) + "\n")


def main():
    # parse CLI args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--language", type=str, default="English", help="The language of the data to be converted")
    parser.add_argument("--phase", type=str, default="Train", help="The phase of the data to be converted")
    parser.add_argument("--path", type=str, default="/content/drive/MyDrive/data", help="The path to read the data from")
    parser.add_argument("--input", type=str, default="structured", help="The input directory")
    parser.add_argument("--output", type=str, default="llm_data", help="The output directory")
    parser.add_argument("--overwrite",action="store_true", help="If you want to overwrite the old file")
    parser.add_argument("--dataset", type=str, default="SemEval16", help="Which dataset you want to convert")

    args = parser.parse_args()

    return create_fine_tuning(args)

if __name__ == "__main__":
    main()