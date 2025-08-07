"""
Code is largely derived from https://github.com/jpeper/GEN_SCL_NAT.git and 
https://github.com/ZubinGou/multi-view-prompting.git
"""
import torch
import json
import os 
import numpy as np

from torch.utils.data import Dataset
from itertools import permutations
from model.generate_data import get_cl_mvp_tasd_data

from transformers import MT5Tokenizer
from model.t5_score import MyMT5ForConditionalGenerationScore

def get_data(data_path):
    sents, labels = [], []

    with open(data_path, "r",  encoding="utf-8") as file:
        data = json.load(file)

        for instance in data:
            words = instance.get("text")
            sents.append(words.split())
            labels.append(instance.get("labels"))

    return sents, labels
    
def get_transformed_io(data_path, task, data_type, top_k, args, llm=False):
    """
    The main function to transform input & target according to the task
    """
    sents, labels = get_data(data_path)
    if task.startswith('cl_mvp_tasd'):
        inputs, sorted_labels = get_cl_mvp_tasd_data(sents, labels, task)

        new_inputs, targets = get_para_targets(inputs, sorted_labels, top_k, data_type, args, llm=llm)

        print(len(inputs), len(new_inputs), len(targets))
        
        if llm: return inputs, targets
        return new_inputs, targets, labels
    else:
        raise NotImplementedError

class ABSADataset(Dataset):
    def __init__(self, tokenizer, task, top_k, args, data_type, max_len=256, data_path=None):
        # './data/rest16/train.txt'
        self.data_path = data_path
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.task = task
        self.args = args
        self.top_k = top_k
        self.data_type = data_type
        self.inputs = []
        self.targets = []
        self.contrastive_labels = {}
        self.sentence_strings = []
        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, 
                "target_ids": target_ids, "target_mask": target_mask}

    def get_raw_labels(self):
        results = get_transformed_io(self.data_path, self.task, self.data_type, self.top_k, self.args)
        return results
        
    def _build_examples(self):

        inputs, targets = get_transformed_io(self.data_path, self.task,self.data_type, self.top_k, self.args)
        self.sentence_strings = inputs
        for i in range(len(inputs)):
            # change input and target to two strings
            input = ' '.join(inputs[i])
            target = targets[i]

            tokenized_input = self.tokenizer.batch_encode_plus(
              [input], max_length=self.max_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
              [target], max_length=self.max_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )
            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)

class GenSCLNatDataset(ABSADataset):

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        sentiment_label = torch.tensor(self.contrastive_labels['sentiment'][index])
        aspect_label = torch.tensor(self.contrastive_labels['aspect'][index])
        
        return {"source_ids": source_ids,
                "source_mask": src_mask, 
                "target_ids": target_ids,
                "target_mask": target_mask,
                'sentiment_labels': sentiment_label,
                'aspect_labels': aspect_label,
                }

    def _build_examples(self):

        inputs, targets, labels = get_transformed_io(self.data_path, self.task, self.data_type, self.top_k, self.args)
        
        self.sentence_strings = inputs
        for i in range(len(inputs)):
            # change input and target to two strings

            input = ' '.join(inputs[i])
            target = targets[i]
            if isinstance(targets[i], list):
                target = " ".join(targets[i])

            tokenized_input = self.tokenizer.batch_encode_plus(
              [input], max_length=self.max_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
              [target], max_length=self.max_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)

        def get_sentiment_labels(labels_in):
            sentiment_dict = {
                'negative': 0,
                'neutral': 1,
                'positive': 2,
                'mixed': 3
            }
            sentiment_labels = []
            for ex in labels_in:
                label = list(set([triplet.get("p") for triplet in ex]))
                if len(label) == 1:
                    label = sentiment_dict[label[0]]
                else:
                    label = sentiment_dict['mixed']
                assert label in [0,1,2,3]
                sentiment_labels.append(label)
            from collections import Counter
            print("Sentiment distribution")
            print(Counter(sentiment_labels))
            return sentiment_labels

        def get_aspect_labels(labels_in):
            aspect_dict = {
                'NULL': 0,
                'EXPLICIT': 1,
                'BOTH': 2,
            }
            aspect_labels = []
            for ex in labels_in:
                aspects = set([triplet.get("a") for triplet in ex])

                if 'NULL' not in aspects:
                    label = aspect_dict['EXPLICIT']
                else:
                    if len(aspects) == 1:
                        label = aspect_dict['NULL']
                    else:
                        label = aspect_dict['BOTH']

                aspect_labels.append(label)
            return aspect_labels
        
        orig_sent = get_sentiment_labels(labels) 
        orig_asp  = get_aspect_labels(labels)

        self.contrastive_labels['sentiment'] = [
            lab for lab in orig_sent for _ in range(self.top_k)
        ]
        self.contrastive_labels['aspect'] = [
            lab for lab in orig_asp  for _ in range(self.top_k)
        ]

def get_para_targets(sents, labels, top_k, data_type, args, llm=False):
    """
    Obtain the target sentence under the paraphrase paradigm
    """
    targets = []
    new_sents = []

    top_k = min(5, top_k)

    if llm:
        optim_orders = ["[A] [C] [P]"]
    else:
        optim_orders = get_orders(args, data_type, sents, labels)[:top_k]

    for i in range(len(sents)):
        label = labels[i]
        cur_sent = sents[i]

        trip_list = []
        for _tuple in label:
            at, ac, sp = _tuple
                
            if at and at.lower() == 'null':
                at = 'it'

            element_dict = {"[A]": at, "[C]": ac, "[P]": sp}
            token_end = 3

            element_list = []
            for key in optim_orders[0].split(" "):
                element_list.append("{} {}".format(key, element_dict[key]))

            x = permutations(element_list)
            permute_object = {}
            for each in x:
                order = []
                content = []
                for e in each:
                    order.append(e[0:token_end])
                    content.append(e[token_end:])
                order_name = " ".join(order)
                content = " ".join(content)
                permute_object[order_name] = [content, " ".join(each)]

            trip_list.append(permute_object)

        for o in optim_orders:
            tar = []
            for each_q in trip_list:
                tar.append(each_q[o][1])

            targets.append(" [SSEP] ".join(tar))
            # add prompt
            new_sent = add_prompt(cur_sent, o.split(), args)
            new_sents.append(new_sent)

    return new_sents, targets

def get_orders(args, data_type, sents, labels):
    path_orders = f"{args.ordering}/{args.dataset}-{args.language}.json"

    if not os.path.exists(args.ordering):
        os.makedirs(args.ordering, exist_ok=True)

    if data_type == "Train" and not os.path.exists(path_orders):
        print(f"\nFinding optimal orders from test set {args.dataset}-{args.language}.\n")

        device = torch.device('cuda:0')
        tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")
        model = MyMT5ForConditionalGenerationScore.from_pretrained(
            "google/mt5-base").to(device)
        optim_orders_all = choose_best_order_global(sents, labels, model,
                                                tokenizer, device)
        
        with open(path_orders, 'w', encoding="utf-8") as f:
            json.dump(optim_orders_all, f)
    else:
        with open(path_orders, 'r', encoding="utf-8") as f:
            optim_orders_all = json.load(f)

    return optim_orders_all


def choose_best_order_global(sents, labels, model, tokenizer, device):
    q =  ["[A]", "[C]", "[P]"]
    all_orders = permutations(q)

    all_orders_list = []
    scores = []

    for each_order in all_orders:
        cur_order = " ".join(each_order)
        all_orders_list.append(cur_order)
        scores.append(0)

    for i in range(len(sents)):
        label = labels[i]
        sent = sents[i]

        trip_list = []
        for _tuple in label:
            at, ac, sp = _tuple
                
            if at and at.lower() == 'null':
                at = 'it'

            element_dict = {"[A]": at, "[C]": ac, "[P]": sp}
            element_list = []
            for key in q:
                element_list.append("{} {}".format(key, element_dict[key]))

            x = permutations(element_list)

            permute_object = {}
            for each in x:
                order = []
                content = []
                for e in each:
                    order.append(e[0:3])
                    content.append(e[3:])
                order_name = " ".join(order)
                content = " ".join(content)
                permute_object[order_name] = [content, " ".join(each)]

            trip_list.append(permute_object)

        order_scores = order_scores_function(trip_list, sent, model, tokenizer,
                                             device)
        for e in order_scores:
            index = all_orders_list.index(e)
            scores[index] += order_scores[e]['entropy']

    indexes = np.argsort(np.array(scores))  # [::-1]
    returned_orders = []
    for i in indexes:
        returned_orders.append(all_orders_list[i])

    print("Orders:", returned_orders)
    return returned_orders

def order_scores_function(trip_list, cur_sent, model, tokenizer, device):
    q =  ["[A]", "[C]", "[P]"]

    all_orders = permutations(q)
    all_orders_list = []

    all_targets = []
    all_inputs = []
    cur_sent = " ".join(cur_sent)
    for each_order in all_orders:
        cur_order = " ".join(each_order)
        all_orders_list.append(cur_order)
        cur_target = []
        for each_q in trip_list:
            cur_target.append(each_q[cur_order][0])

        all_inputs.append(cur_sent)
        all_targets.append(" ".join(cur_target))

    tokenized_input = tokenizer.batch_encode_plus(all_inputs,
                                                  max_length=200,
                                                  padding="max_length",
                                                  truncation=True,
                                                  return_tensors="pt")
    tokenized_target = tokenizer.batch_encode_plus(all_targets,
                                                   max_length=200,
                                                   padding="max_length",
                                                   truncation=True,
                                                   return_tensors="pt")

    target_ids = tokenized_target["input_ids"].to(device)

    target_ids[target_ids[:, :] == tokenizer.pad_token_id] = -100
    outputs = model(
        input_ids=tokenized_input["input_ids"].to(device),
        attention_mask=tokenized_input["attention_mask"].to(device),
        labels=target_ids,
        decoder_attention_mask=tokenized_target["attention_mask"].to(device))

    loss, entropy = outputs[0]
    results = {}
    for i, _ in enumerate(all_orders_list):
        cur_order = all_orders_list[i]
        results[cur_order] = {"loss": loss[i], "entropy": entropy[i]}
    # print(best_quad)
    return results

def add_prompt(sent, orders, args):
    # add ctrl_token
    if args.ctrl_token == "none":
        pass
    elif args.ctrl_token == "post":
        sent = sent + orders
    elif args.ctrl_token == "pre":
        sent = orders + sent
    else:
        raise NotImplementedError
    return sent