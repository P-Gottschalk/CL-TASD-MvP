from openai import OpenAI
import argparse
import os
import time

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def finetuning(args):
    train_path = f"{args.path}/{args.dataset}_Restaurants_Train_{args.language}.jsonl"
    dev_path = f"{args.path}/{args.dataset}_Restaurants_Dev_{args.language}.jsonl"

    train_file = client.files.create(file=open(train_path, "rb"), purpose="fine-tune")
    print(f"Training file id: {train_file.id}")

    val_file = client.files.create(file=open(dev_path, "rb"), purpose="fine-tune")
    print(f"Validation file id: {val_file.id}")

    finetune_job = client.fine_tuning.jobs.create(training_file=train_file.id, validation_file=val_file.id, model=args.model)

    while True:
        job_status = client.fine_tuning.jobs.retrieve(finetune_job.id)
        print("Job status:", job_status.status)

        if job_status.status in ["succeeded", "failed", "cancelled"]:
            break

        time.sleep(30)

    if job_status.status == "succeeded":
        model_id = job_status.fine_tuned_model
        print(f"Fine-tuned model id: {model_id}")
    
        output_path = f"{args.path}/Model_IDs.txt"
        with open(output_path, 'a', encoding="utf-8") as file:
            file.write(f"\nModel ID; {args.dataset}; {args.language}; {model_id}")
        
        return model_id
    else:
        print("Fine-tuning has failed or was cancelled.")
        print(f"Final status: {job_status.status}")

def main():
    # parse CLI args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--language", type=str, default="English", help="The language of the data used for fine-tuning")
    parser.add_argument("--dataset", type=str, default="SemEval16", help="Which dataset used for fine-tuning")
    parser.add_argument("--path", type=str, default="/content/drive/MyDrive/data/llm_data", help="The path to read the data from")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini-2025-04-14", help="The model to be used.")

    args = parser.parse_args()

    finetuning(args)

if __name__ == "__main__":
    main()