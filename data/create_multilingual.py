import argparse
import os
import json
import random

def create_multilingual(args):
    filename_multilingual = f"{args.dataset}_Restaurants_{args.phase}_Multilingual"

    if args.dataset == "MABSA":
        if not args.full_data:
            filename_multilingual = filename_multilingual + "_Small"
        if args.size_limit:
            filename_multilingual = filename_multilingual + "_Limited"

    output_path = f"{args.path}/{filename_multilingual}.json"

    if os.path.isfile(output_path) and not args.overwrite:
        print(f"Found cleaned file at {output_path}")
        with open(output_path, "r") as file:
            return json.load(file)
        
    full_dataset = []

    if not args.size_limit:
        for language in args.languages_semeval:
            input_path = f"{args.path}/{args.dataset}_Restaurants_{args.phase}_{language}.json"

            with open(input_path, "r") as file:
                full_dataset.extend(json.load(file))


        if args.dataset == "MABSA" and args.full_data:
            for language in args.languages_mabsa:
                input_path = f"{args.path}/{args.dataset}_Restaurants_{args.phase}_{language}.json"

                with open(input_path, "r") as file:
                    full_dataset.extend(json.load(file))
    else:
        sequence, data = create_sequence_data(args)

        for index, item in enumerate(sequence):
            full_dataset.append(data.get(item)[index])

    with open(output_path, "w") as file:
        json.dump(full_dataset, file, ensure_ascii=False, indent=2)
    
    print(f"Created Multilingual dataset of size: {len(full_dataset)}")

    return full_dataset


def create_sequence_data(args):
    languages = []
    languages.extend(args.languages_semeval)

    if args.full_data:
        languages.extend(args.languages_mabsa)

    size = 0
    data = {}
    for language in languages:
        input_path = f"{args.path}/{args.dataset}_Restaurants_{args.phase}_{language}.json"
        with open(input_path, "r") as file:
            temp_data = json.load(file)
            data[language] = temp_data
            size = len(temp_data)

    count = size // len(languages)
    remainder = size % len(languages)

    sequence = []

    for index, lang in enumerate(languages):
        num = count + (1 if index < remainder else 0)
        sequence.extend([lang] * num)

    random.shuffle(sequence)
    return sequence, data


def main():
    # parse CLI args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--phase", type=str, default="Train", help="The phase of the data to be converted")
    parser.add_argument("--path", type=str, default="/content/drive/MyDrive/data/structured", help="The path to read the data from")
    parser.add_argument("--overwrite", action="store_true", help="If you want to overwrite the old file")
    parser.add_argument("--dataset", type=str, default="SemEval16", help="Which dataset you want to convert: [SemEval16, MABSA]")
    parser.add_argument("--full_data", action="store_true", help="If you want the full MABSA set or only the 5 languages also in SemEval16")
    parser.add_argument("--size_limit", action="store_true", help="Impose an artificial size limit.")
    parser.add_argument("--dev_seed", type=int, default=42, help="The seed for the multilingual set.")

    args = parser.parse_args()

    languages_semeval = [
        "English", 
        "French", 
        "Spanish", 
        "Dutch",
        "Turkish"
        ]
    languages_mabsa = [
        "Danish",
        "German",
        "Indonesian",
        "Portuguese",
        "Slovak",
        "Swahili",
        "Swedish",
        "Vietnamese"
    ]

    args.languages_semeval = languages_semeval
    args.languages_mabsa = languages_mabsa

    random.seed(args.dev_seed)

    dataset = create_multilingual(args)

if __name__ == "__main__":
    main()