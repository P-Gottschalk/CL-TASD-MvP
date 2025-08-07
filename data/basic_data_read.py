import argparse
import os
import ast
import json
import re

def read_and_filter(language, phase, path, input, output, overwrite, dataset):
    list_data = []

    filename = f"{dataset}_Restaurants_{phase}_{language}"

    input_path = f"{path}/{input}/{filename}.txt"
    output_path = f"{path}/{output}/{filename}.json"

    if os.path.isfile(output_path) and not overwrite:
        print(f"Found cleaned file at {output_path}")
        with open(output_path, "r") as file:
            return json.load(file)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    count_lines = 0
    count_total = 0
    count_implicit = 0

    with open(input_path, 'r', encoding='UTF-8') as file_txt:

        for line in file_txt:
            count_lines += 1

            line = line.strip()
            if line != '':
                instance, labels = line.split('####')

            try: 
                labels = ast.literal_eval(labels)
            except:
                labels = re.sub(r"(?<=\[|\s|,)'(.*?)'(?=,|\])", r'"\1"', labels)
                labels = ast.literal_eval(labels)

            instance = re.sub(r'\s{2,}', ' ', re.sub(r'([.,!?;:()"-])(?=\S)|(?<=\S)([.,!?;:()"-])', r' \1\2 ', instance)).strip()

            if labels:
                sentiment_dictionary = dict()
                labels_list = []

                sentiment_dictionary["text"] = instance

                for label_set in labels:
                    count_total += 1

                    set_dict = dict()

                    set_dict["a"] = label_set[0]
                    set_dict["c"] = label_set[1]
                    set_dict["p"] = label_set[2]

                    if label_set[0].lower() == "null":
                        count_implicit += 1

                    labels_list.append(set_dict)

                sentiment_dictionary["labels"] = labels_list

                list_data.append(sentiment_dictionary)

    with open(output_path, "w") as file: 
        json.dump(list_data, file, ensure_ascii=False, indent=2)

    print(f"\n# Sentences: {count_lines} \n# Total Aspects (Implicit Aspects): {count_total} ({count_implicit})")

    return list_data

def main():
    # parse CLI args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--language", type=str, default="English", help="The language of the data to be converted")
    parser.add_argument("--phase", type=str, default="Train", help="The phase of the data to be converted")
    parser.add_argument("--path", type=str, default="/content/drive/MyDrive/data", help="The path to read the data from")
    parser.add_argument("--input", type=str, default="txt_raw", help="The input directory")
    parser.add_argument("--output", type=str, default="structured", help="The output directory")
    parser.add_argument("--overwrite",action="store_true", help="If you want to overwrite the old file")
    parser.add_argument("--dataset", type=str, default="SemEval16", help="Which dataset you want to convert")

    args = parser.parse_args()

    language: str = args.language
    phase: str = args.phase
    path:  str = args.path
    input: str = args.input
    output: str = args.output
    overwrite: bool = args.overwrite
    dataset: str = args.dataset

    return read_and_filter(language=language, phase=phase, path=path, input=input, output=output, overwrite=overwrite, dataset=dataset)


if __name__ == "__main__":
    main()