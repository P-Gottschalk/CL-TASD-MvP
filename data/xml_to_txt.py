import argparse
import os
import re
import xml.etree.ElementTree as ElementTree
import random

def txt_xml_conversion(language: str, phase: str, path: str, input: str, output: str, overwrite: bool):
    filename = f"SemEval16_Restaurants_{phase}_{language}"

    input_path = f"{path}/{input}/{filename}.xml" 
    output_path = f"{path}/{output}/{filename}_NoSplit.txt" if phase == "Train" else f"{path}/{output}/{filename}.txt"

    if os.path.isfile(output_path) and not overwrite:
        print(f"Found cleaned file at {output_path}")
        return open(output_path, "r")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    tree = ElementTree.parse(input_path)

    with open(output_path, 'w') as txt_file:
        for sentence in tree.findall(".//sentence"):
            text = sentence.find("text").text
            if text is None:
                continue

            list_opinions = []
            for opinion in sentence.findall(".//Opinions/Opinion"):
                temp_list_opinion = [opinion.get("target"), (re.sub(r"#", " ", opinion.get("category"))).lower(), opinion.get("polarity")]
                list_opinions.append(temp_list_opinion)
                
            string_opinion = f"####{str(list_opinions)}"
            string = text + string_opinion

            txt_file.write(string + "\n")
    
    return open(output_path, "r")

def create_dev_set(language: str, path: str, input: str, dev_seed: int, dev_size: float, overwrite: bool):
    random.seed(dev_seed)

    filename_train = f"SemEval16_Restaurants_Train_{language}"
    filename_dev = f"SemEval16_Restaurants_Dev_{language}"

    input_path = f"{path}/{input}/{filename_train}_NoSplit.txt"

    output_path_train = f"{path}/{input}/{filename_train}.txt"
    output_path_dev = f"{path}/{input}/{filename_dev}.txt"

    if os.path.isfile(output_path_train) and os.path.isfile(output_path_dev) and not overwrite:
        print(f"Found cleaned files for SemEval16 for {language}")
        return open(output_path_dev, "r"), open(output_path_train, "r")

    with open(input_path, 'r') as file:
        with open(output_path_train, 'w') as train:
            with open(output_path_dev, 'w') as dev:
                for line in file:
                    if random.random() < dev_size:
                        dev.write(line)
                    else:
                        train.write(line)

    return open(output_path_dev, 'r'), open(output_path_train, 'r')

def main():
    # parse CLI args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--language", type=str, default="English", help="The language of the datat to be converted")
    parser.add_argument("--phase", type=str, default="Train", help="The phase of the data to be converted")
    parser.add_argument("--path", type=str, default="/content/drive/MyDrive/data", help="The path to read the data from")
    parser.add_argument("--input", type=str, default="sem_eval_raw", help="The input directory")
    parser.add_argument("--output", type=str, default="txt_raw", help="The output directory")
    parser.add_argument("--overwrite", action="store_true", help="If you want to overwrite the old file")
    parser.add_argument("--dev_seed", type=int, default=42, help="The seed to create the dev set for SemEval16")
    parser.add_argument("--dev_size", type=float, default=0.1, help="Fraction of the training data to be converted into a dev set.")

    args = parser.parse_args()

    language: str = args.language
    phase: str = args.phase
    path:  str = args.path
    input: str = args.input
    output: str = args.output
    overwrite: bool = args.overwrite
    seed: int = args.dev_seed
    size: float = args.dev_size

    txt = txt_xml_conversion(language=language, phase=phase, path=path, input=input, output=output, overwrite=overwrite)

    if phase == "Train":
        dev, train = create_dev_set(language=language, path=path, input=output, dev_seed=seed, dev_size=size, overwrite=overwrite)

if __name__ == "__main__":
    main()
