import argparse
import os
import json
import pandas as pd

def create_dataframe(args, which_type):
    suffix_map = {
        "best_ckpt": "-ckpt.json",
        "last_5":   "-5ckpt.json",
        "full_model": ".json",
    }
    try:
        suffix = suffix_map[which_type]
    except KeyError:
        raise ValueError(f"Unknown type: {which_type!r}")

    columns = ["Full Model", "MVP only", "CL only", "Base MT5"]
    df = pd.DataFrame(index=pd.Index([], name="Setting"), columns=columns)

    all_languages = list(args.languages) + ["English"]

    def process_mode(mode_name, langs):
        for col in columns:
            mvp = 5 if col in ("Full Model", "MVP only") else 1
            contr = "-sent-aspect" if col in ("Full Model", "CL only") else ""

            for lang in langs:

                if mode_name == "Unilingual":
                    folder_prefix = lang
                elif mode_name == "Cross-Lingual-ENG":
                    folder_prefix = "English"
                else:
                    folder_prefix = mode_name

                for extra in ("", "-CD"):
                    filename = (f"{folder_prefix}_{args.dataset}/"f"results-{lang}-{args.dataset}-"f"{args.num_train_epochs}epochs-"f"{mvp}mvp-{args.weight_decay}-"f"{args.warmup_steps}-"f"{args.hidden_layers}{extra}{contr}"f"{suffix}")

                    full_path = os.path.join(args.path, filename)
                    row_idx = f"{mode_name}{extra}: {lang}"

                    if os.path.exists(full_path):
                        with open(full_path, 'r', encoding="utf-8") as f:
                            data = json.load(f)
                        score = data.get("performance_metrics", {}).get("f1", None)
                    else:
                        score = None
                    df.loc[row_idx, col] = score

    process_mode("Unilingual", all_languages)

    if args.multilingual:
        process_mode("Multilingual", all_languages)

    if args.multilingual_limited:
        process_mode("Multilingual_Limited", all_languages)

    if args.multilingual_small:
        process_mode("Multilingual_Small", all_languages)

    if args.multilingual_small_limited:
        process_mode("Multilingual_Small_Limited", all_languages)

    if args.english:
        process_mode("Cross-Lingual-ENG", args.languages)

    return df.sort_index()


def create_dataframe_llm(args):
    path_to_results = f"{args.path_llm}/0_{args.dataset}_Results.json"

    with open(path_to_results,'r',encoding='utf-8') as file:
        data = json.load(file)

    df = pd.DataFrame(index=pd.Index([], name="Setting"))

    for key in data:
        f1_score = data.get(key).get("f1")

        key_split = key.split("_")

        train_lang = "_".join(key_split[0:-2])
        if "_" in train_lang:
            train_lang = "0_" + train_lang
        test_lang = key_split[-2]
        type_model = key_split[-1]

        if "Multilingual" in train_lang:
            index = f"{train_lang}_{test_lang}"
        elif train_lang != test_lang:
            index = f"Cross-lingual_{train_lang}_{test_lang}"
        else:
            index = f"Unilingual_{train_lang}_{test_lang}"
        
        df.loc[index, type_model] = f1_score

    return df.sort_index()
    
def main():
    # parse CLI args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--path", type=str, default="/content/drive/MyDrive/data/models", help="The path to read the data from")
    parser.add_argument("--languages", nargs="+", default=["French", "Spanish", "Dutch", "Turkish"], help="List of languages")

    parser.add_argument("--full_model", action='store_true', help='Include the full model results')
    parser.add_argument("--last_5", action='store_true', help='Include the results from the last 5 epochs')
    parser.add_argument("--best_ckpt", action='store_true', help='Include the results from the best checkpoint')

    parser.add_argument("--english", action='store_true', help='English models have been trained')
    parser.add_argument("--multilingual", action='store_true', help='Multilingual models have been trained')
    parser.add_argument("--multilingual_limited", action='store_true', help='Multilingual limited models have been trained')
    parser.add_argument("--multilingual_small", action='store_true', help='Multilingual small models have been trained')
    parser.add_argument("--multilingual_small_limited", action='store_true', help='Multilingual small limited models have been trained')

    parser.add_argument("--hidden_layers", default=1024, type=int, help="Size of the Seq2Seq model")
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--warmup_steps", default=0.1, type=float)
    parser.add_argument("--dataset", type=str, default="SemEval16",  help="Which dataset you want to use")
    parser.add_argument("--num_train_epochs", default=20, type=int, help="Total number of training epochs to perform.")

    parser.add_argument("--llm", action='store_true', help="Create tables for LLMs")
    parser.add_argument("--path_llm", type=str, default="/content/drive/MyDrive/data/llm", help="The path to the LLM folder")

    args = parser.parse_args()

    output_dir = os.path.join(args.path, "combined_results")
    os.makedirs(output_dir, exist_ok=True)

    if not args.llm:
        for flag, which in [(args.full_model, "full_model"), (args.last_5, "last_5"), (args.best_ckpt, "best_ckpt")]:
            if flag:
                df = create_dataframe(args, which)
                out_file = os.path.join(output_dir, f"results-{args.dataset}-{args.num_train_epochs}-{args.hidden_layers}-{args.weight_decay}-{args.warmup_steps}-{which}.xlsx")
                df.to_excel(out_file, index=True)

                print(f"Wrote results to {out_file}")
    else:
        df = create_dataframe_llm(args)
        out_file = os.path.join(output_dir, f"results-{args.dataset}-LLM.xlsx")
        df.to_excel(out_file, index=True)

        print(f"Wrote results to {out_file}")

if __name__ == "__main__":
    main()


