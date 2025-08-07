import argparse
import os
import sys
import logging
from tqdm import tqdm
from collections import Counter
from functools import partial
import json

sys.path.append('/content/CL-TASD-MvP')
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from torch import nn
from torch.nn.functional import normalize
import torch
from torch.utils.data import DataLoader
from transformers import Adafactor
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from utils.sup_con_loss import SupConLoss
from pytorch_lightning.callbacks import ModelCheckpoint
import shutil

import huggingface_hub.utils._validators as _validators

_original_validate = _validators.validate_repo_id
def _validate_repo_id_override(repo_id: str) -> str:
    if os.path.exists(repo_id):
        return repo_id
    return _original_validate(repo_id)

_validators.validate_repo_id = _validate_repo_id_override

from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from transformers import get_linear_schedule_with_warmup

from pathlib import Path

from data_utils import GenSCLNatDataset
from eval_utils import compute_scores, extract_spans_para
from utils.const import *

logger = logging.getLogger(__name__)

DEVICE = f'cuda:{0}'
torch.set_float32_matmul_precision("medium")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--task", default='cl_mvp_tasd', type=str, required=True,
                        help="The name of the task")
    parser.add_argument("--language", type=str, default="English", 
                        help="The language of the training data")
    parser.add_argument("--test_language", type=str, default="English",
                        help="The language of the testing data")
    parser.add_argument("--path", type=str, default="/content/drive/MyDrive/data/structured", 
                        help="The path to read the data from")
    parser.add_argument("--ordering", type=str, default="/content/drive/MyDrive/data/orders",
                        help="The path to find the respective model orders")
    parser.add_argument("--constrained_decoding", type=str, default="/content/drive/MyDrive/data/constrained_decoding",
                        help="The path for constrained decoding files")
    parser.add_argument("--do_con_dec", action='store_true', help='Constrained decoding when evaluating')
    parser.add_argument("--do_contr_aspect", action='store_true', help='Activate aspect contrastive learning')
    parser.add_argument("--do_contr_sent", action='store_true', help='Activate sentiment contrastive learning')
    parser.add_argument("--dataset", type=str, default="SemEval16", 
                        help="Which dataset you want to use")
    parser.add_argument("--model_name_or_path", default='google/mt5-large', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--hidden_layers", default=1024, type=int,
                        help="Size of the Seq2Seq model")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_direct_eval", action='store_true', 
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_inference", action='store_true', 
                        help="Whether to run inference with trained checkpoints")
    parser.add_argument("--do_inference_best_ckpt", action='store_true', 
                        help="Whether to run inference with best checkpoints")
    parser.add_argument("--do_inference_best_5", action='store_true', 
                        help="Whether to run inference with best checkpoint out of the last 5")
    parser.add_argument("--ctrl_token",
                        default="post",
                        choices=["post", "pre", "none"],
                        type=str)
    parser.add_argument("--top_k", default=5, type=int)

    # other parameters
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--n_gpu", default=1, type=int)
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--num_train_epochs", default=30, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # training details
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)
    parser.add_argument("--output_folder", type=str, 
    default="/content/drive/MyDrive/data/models")
    parser.add_argument("--model_prefix", type=str, required=True)
    parser.add_argument("--num_beams", type=int, required=True)
    parser.add_argument("--early_stopping", type=int, default=0)
    parser.add_argument("--cont_loss", type=float, default=0.0)
    parser.add_argument("--cont_temp", type=float, default=0.1)

    args = parser.parse_args()

    # create output folder if needed
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    
    params = [['beams', str(args.num_beams)],
              ['wd', str(args.weight_decay)],
              ['max_epochs', str(args.num_train_epochs)],
              ['es', str(args.early_stopping)],
              ['acc', str(args.gradient_accumulation_steps)],
              ['lr', str(args.learning_rate)],
              ['cont_loss', str(args.cont_loss)],
              ['cont_temp', str(args.cont_temp)],
              ['seed', str(args.seed)]]

    # the model path is the prefix
    if (args.do_inference or args.do_inference_best_ckpt or args.do_inference_best_5) and not args.do_train:
        output_fold = args.model_prefix
        print(output_fold)
    else:
        # dump params as part of folder_path
        params = "I".join([elt for elts in params for elt in elts])
        output_fold = "I".join([args.dataset, args.task,args.model_name_or_path, params, args.model_prefix, str(args.top_k), str(args.warmup_steps)])
        if args.do_contr_sent:
            output_fold = "I".join([output_fold, "sent"])
        if args.do_contr_aspect:
            output_fold = "I".join([output_fold, "aspect"])
        print(output_fold)

    output_dir = f"{args.output_folder}/{output_fold}"

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder, exist_ok=True)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir

    return args

def get_dataset_training(tokenizer, phase, args):
    data_path = f"{args.path}/{args.dataset}_Restaurants_{phase}_{args.language}.json"
    return GenSCLNatDataset(tokenizer=tokenizer, data_path=data_path ,top_k=args.top_k, data_type=phase, max_len=args.max_seq_length, task=args.task, args = args)

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(args.hidden_layers, args.hidden_layers)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x, attention_mask):
        last_state = torch.mul(x, attention_mask.unsqueeze(-1))
        features_summed = torch.sum(last_state, dim=1)
        dropped = self.dropout(features_summed)
        return torch.stack((self.layer_1(features_summed), self.layer_1(dropped)), 1)

class MT5FineTuner(pl.LightningModule):
    def __init__(self, hparams, tfm_model, tokenizer, cont_model, as_model, cat_model):
        super(MT5FineTuner, self).__init__()
        self.hparams.update(vars(hparams))
        self.model = tfm_model
        self.cont_model = cont_model
        self.as_model = as_model
        self.cat_model = cat_model
        self.tokenizer = tokenizer
        self._val_preds = []
        self._val_targets = []

    def is_logger(self):
        return True

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        main_pred = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True,
        )
        last_state = main_pred.encoder_last_hidden_state
        cont_pred = self.cont_model(last_state, attention_mask)
        as_pred = self.as_model(last_state, attention_mask)
        masked_last_state = torch.mul(last_state, attention_mask.unsqueeze(-1))
        pooled_encoder_layer = torch.sum(masked_last_state, dim=1)
        pooled_encoder_layer = normalize(pooled_encoder_layer, p=2.0, dim=1)
        return main_pred, cont_pred, as_pred, pooled_encoder_layer

    def _step(self, batch):
        lm_labels = torch.clone(batch["target_ids"])
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs, cont_pred, as_pred, pooled_encoder_layer = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask'],
        )

        criterion = SupConLoss(loss_scaling_factor=self.hparams.cont_loss, temperature=self.hparams.cont_temp)
        sentiment_labels = batch['sentiment_labels']
        aspect_labels = batch['aspect_labels']

        sentiment_contrastive_loss = 0
        aspect_contrastive_loss = 0

        if self.hparams.do_contr_sent:
            cont_summed = cont_pred
            cont_normed = normalize(cont_summed, p=2.0, dim=2)  
            sentiment_contrastive_loss = criterion(cont_normed, sentiment_labels)
        if self.hparams.do_contr_aspect:
            as_summed = as_pred
            as_normed = normalize(as_summed, p=2.0, dim=2)
            aspect_contrastive_loss = criterion(as_normed, aspect_labels)

        loss = outputs[0] + sentiment_contrastive_loss + aspect_contrastive_loss
        return loss, outputs

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")
    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")
    def training_step(self, batch, batch_idx):
        loss, _ = self._step(batch)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    
    def on_validation_epoch_start(self):
        self._val_preds = []
        self._val_targets = []


    def on_validation_epoch_end(self):
        top_k = self.hparams.top_k

        targets = self._val_targets
        preds = self._val_preds

        if top_k > 1:
            targets = targets[::top_k]

            processed_preds = []

            for i in range(0, len(targets)):
                o_idx = i * top_k
                multi_outputs = preds[o_idx:o_idx + top_k]

                all_triplets = []
                for s in multi_outputs:
                    all_triplets.extend(
                        extract_spans_para(seq=s, seq_type='pred'))
                    
                output_triplets = []
                counter = dict(Counter(all_triplets))
                for trip, count in counter.items():
                    if count >= len(multi_outputs) / 2:
                        output_triplets.append(trip)

                output = []
                for t in output_triplets:
                    ac, at, sp = t
                    output.append(f"[A] {at} [P] {sp} [C] {ac}")

                output_str = " [SSEP] ".join(
                    output) if output else multi_outputs[0]

                processed_preds.append(output_str)
            
            preds = processed_preds
        
        scores, _, _ = compute_scores(preds, targets)

        epoch_f1 = torch.tensor(scores["f1"], dtype=torch.float)
        self.log("val_f1", epoch_f1, prog_bar=True, on_epoch=True)


    def evaluate(self, batch, stage=None):
        outs = self.model.generate(input_ids=batch['source_ids'],
                                   attention_mask=batch['source_mask'],
                                   max_length=self.hparams.max_seq_length,
                                   return_dict_in_generate=True,
                                   output_scores=True,
                                   num_beams=1)
        dec = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in outs.sequences
        ]
        target = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in batch["target_ids"]
        ]

        if stage == "val":
            self._val_preds.extend(dec)
            self._val_targets.extend(target)
    
        loss, _ = self._step(batch)

        if stage:
            self.log(f"{stage}_loss",
                     loss,
                     prog_bar=True,
                     on_step=False,
                     on_epoch=True)

    def configure_optimizers(self):
        model = self.model
        cont_model = self.cont_model
        as_model = self.as_model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in cont_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in cont_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in as_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in as_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = Adafactor(
            optimizer_grouped_parameters, 
            lr=self.hparams.learning_rate,
            scale_parameter=False,
            relative_step=False,
            weight_decay=self.hparams.weight_decay,
            clip_threshold=1.0,
            eps=(1e-30, 1e-3)
        )
        self.opt = optimizer

        total_steps = self.trainer.estimated_stepping_batches

        if isinstance(self.hparams.warmup_steps, float) and self.hparams.warmup_steps < 1:
            num_warmup_steps = int(self.hparams.warmup_steps * total_steps)
        else:
            num_warmup_steps = int(self.hparams.warmup_steps)
            
        print(f"Using total_steps = {total_steps}, num_warmup_steps = {num_warmup_steps}")

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps
        )
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "name": "linear_schedule_with_warmup"
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config
        }

    def train_dataloader(self):
        train_dataset = get_dataset_training(tokenizer=self.tokenizer, phase="Train", args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size,
                                drop_last=True, shuffle=True, num_workers=8)
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset_training(tokenizer=self.tokenizer, phase="Dev", args=self.hparams)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=8)
    
    # Adapted from https://github.com/ZubinGou/multi-view-prompting.git
    def prefix_allowed_tokens_fn(self, source_ids, batch_id, input_ids):
        """
        Constrained Decoding
        # ids = self.tokenizer("text", return_tensors='pt')['input_ids'].tolist()[0]
        """
        if not os.path.exists(f'{args.constrained_decoding}/force_tokens.json'):
            dic = {"cate_tokens":[], "all_tokens":[], "sentiment_tokens":[], 'special_tokens':[]}

            tokenize_res = []
            for w in force_words:
                tokenize_res.extend(self.tokenizer(w, return_tensors='pt')['input_ids'].tolist()[0])
            dic["all_tokens"] = tokenize_res

            tokenize_res = []
            for cat in rest_aspect_cate_list:
                tokenize_res.extend(self.tokenizer(cat, return_tensors='pt')['input_ids'].tolist()[0]) 
            dic["cate_tokens"] = tokenize_res

            sp_tokenize_res = []
            for sp in ['positive', 'neutral', 'negative']:
                sp_tokenize_res.extend(self.tokenizer(sp, return_tensors='pt')['input_ids'].tolist()[0])
            dic['sentiment_tokens'] = sp_tokenize_res

            special_tokens_tokenize_res = []
            for w in ['[A','[S','[C','[P']:
                special_tokens_tokenize_res.extend(self.tokenizer(w, return_tensors='pt')['input_ids'].tolist()[0]) 
            special_tokens_tokenize_res = [r for r in special_tokens_tokenize_res if r != 491]
            dic['special_tokens'] = special_tokens_tokenize_res

            with open(f'{args.constrained_decoding}/force_tokens.json', 'w',  encoding="utf-8") as f:
                json.dump(dic, f, indent=4)

        with open(f'{args.constrained_decoding}/force_tokens.json', 'r',  encoding="utf-8") as f:
            force_tokens = json.load(f)

        to_id = {
            'AT': [357],
            'SP': [559],
            'AC': [424],
            'S': [399],
            'SEP': [155719],
            '[':  [491],
            ']':  [439],
            'it': [609],
            'null': [259, 1181],  
        }

        left_brace_index = (input_ids == to_id['['][0]).nonzero()
        right_brace_index = (input_ids == to_id[']'][0]).nonzero()
        num_left_brace = len(left_brace_index)
        num_right_brace = len(right_brace_index)
        last_right_brace_pos = right_brace_index[-1][
            0] if right_brace_index.nelement() > 0 else -1
        last_left_brace_pos = left_brace_index[-1][
            0] if left_brace_index.nelement() > 0 else -1
        cur_id = input_ids[-1]

        if cur_id in to_id['[']:
            return force_tokens['special_tokens']
        elif cur_id in to_id['AT'] + to_id['SEP'] + to_id['SP'] + to_id['AC']:  
            return to_id[']']  
        elif cur_id in to_id['S']:  
            return to_id['SEP'] 

        # get cur_term
        if last_left_brace_pos == -1:
            return to_id['['] + [1]   # start of sentence: [
        elif (last_left_brace_pos != -1 and last_right_brace_pos == -1) \
            or last_left_brace_pos > last_right_brace_pos:
            return to_id[']']  # ]
        else:
            cur_term = input_ids[last_left_brace_pos + 1].item()

        ret = []
        if cur_term in to_id['SP']:  # SP
            ret = force_tokens['sentiment_tokens'] + [259] + to_id[']'] + [1]
        elif cur_term in to_id['AT']:  # AT
            force_list = source_ids[batch_id].tolist()
            force_list.extend(to_id['it'] + [1])  
            ret = force_list  
        elif cur_term in to_id['S']:
            ret = [259] + to_id[']'] + [1]
        elif cur_term in to_id['AC']:  # AC
            ret = force_tokens['cate_tokens']
        elif cur_term != 1 or num_left_brace > 1:
            raise ValueError(cur_term)

        if num_left_brace == num_right_brace:
            ret = set(ret)
            ret.discard(to_id[']'][0]) # remove ]
            for w in force_tokens['special_tokens']:
                ret.discard(w)
            ret = list(ret)
        elif num_left_brace > num_right_brace:
            ret += to_id[']'] 
        else:
            raise ValueError
        
        ret.extend(to_id['['] + [1]) # add [
        return ret


class BestOfLastNEpochs(pl.Callback):
    def __init__(self, dirpath, monitor="val_f1", mode="max", n=5, filename="best-last5"):
        super().__init__()
        self.dirpath = dirpath
        self.monitor = monitor
        self.mode = mode
        self.n = n
        self.filename = filename
        self._records = []
        os.makedirs(self.dirpath, exist_ok=True)

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            return
        epoch = trainer.current_epoch
        val = metrics[self.monitor].item()
        ckpt = trainer._checkpoint_connector.dump_checkpoint()
        self._records.append((epoch, val, ckpt))
        if len(self._records) > self.n:
            self._records.pop(0)

    def on_train_end(self, trainer, pl_module):
        if not self._records:
            return
        if self.mode == "max":
            best = max(self._records, key=lambda x: x[1])
        else:
            best = min(self._records, key=lambda x: x[1])
        _, best_val, best_ckpt = best
        out_path = os.path.join(self.dirpath, f"{self.filename}.ckpt")
        torch.save(best_ckpt, out_path)
        print(f"[BestOfLastNEpochs] Saved best of last {self.n} epochs "
              f"(epoch val_f1={best_val:.4f}) to {out_path}")
        

class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))


class SafeModelCheckpoint(ModelCheckpoint):
    def __init__(self, dirpath: str, filename: str = "best-overall", monitor: str = "val_f1", mode: str = "max"):
        super().__init__(dirpath=dirpath, filename=filename, monitor=monitor, mode=mode, save_top_k=1, save_last=False, save_on_train_epoch_end=False, verbose=True,)
    def _save_checkpoint(self, trainer, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        ckpt = trainer._checkpoint_connector.dump_checkpoint()
        torch.save(ckpt, filepath)


def evaluate(data_loader, model, sents, args):
    """
    Compute scores given the predictions and gold labels and dump to file
    """
    device = torch.device(DEVICE)
    model.model.to(device)

    model.eval()
    model.model.eval()

    outputs, targets = [], []
    top_k = args.top_k

    print()
    print ("Evaluate using Constrained Decoding.") if args.do_con_dec else print("Evaluate without Constrained Decoding.")
    print()

    for batch in tqdm(data_loader):
        if args.do_con_dec:
            outs = model.model.generate(input_ids=batch['source_ids'].to(device), 
                                        attention_mask=batch['source_mask'].to(device), 
                                        max_length=args.max_seq_length*2,
                                        num_beams=args.num_beams,
                                        prefix_allowed_tokens_fn=partial(model.prefix_allowed_tokens_fn,
                                                                        batch['source_ids']))
        else:
            outs = model.model.generate(input_ids=batch['source_ids'].to(device), 
                                        attention_mask=batch['source_mask'].to(device), 
                                        max_length=args.max_seq_length*2,
                                        num_beams=args.num_beams)

        dec = [model.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        target = [model.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]

        outputs.extend(dec)
        targets.extend(target)
    
    if top_k > 1:
        targets = targets[::top_k]

        _outputs = outputs # backup
        outputs = [] # new outputs

        for i in range(0, len(targets)):
            o_idx = i * top_k
            multi_outputs = _outputs[o_idx:o_idx + top_k]

            all_triplets = []
            for s in multi_outputs:
                all_triplets.extend(
                    extract_spans_para(seq=s, seq_type='pred'))
                
            output_triplets = []
            counter = dict(Counter(all_triplets))
            for trip, count in counter.items():
                if count >= len(multi_outputs) / 2:
                    output_triplets.append(trip)

            output = []
            for t in output_triplets:
                ac, at, sp = t
                output.append(f"[A] {at} [P] {sp} [C] {ac}")

            # if no output, use the first path
            output_str = " [SSEP] ".join(
                output) if output else multi_outputs[0]

            outputs.append(output_str)

    scores, all_labels, all_preds = compute_scores(outputs, targets, silent=False)

    labels_counts = Counter([len(l.split('[SSEP]')) for l in outputs])
    print("pred labels count", labels_counts)

    unique_sents = sents[::args.top_k]

    results = {'labels_correct': all_labels, 'labels_pred': all_preds, 'output_pred': outputs, 'output_correct': targets, 'utterances': unique_sents}
    ex_list = []

    for idx in range(len(all_preds)):
        new_dict = {}
        for key in results:
            new_dict[key] = results[key][idx]
        ex_list.append(new_dict)
    
    results = {'performance_metrics': scores, 'examples': ex_list}

    sent = "-sent" if args.do_contr_sent else ""
    aspect = "-aspect" if args.do_contr_aspect else "" 
    con_dec = "-CD" if args.do_con_dec else ""
    if args.do_inference_best_ckpt:
        best_ckpt = "-ckpt"
    elif args.do_inference_best_5:
        best_ckpt = "-5ckpt"
    else:
        best_ckpt = ""

    json.dump(results, open(f"{args.output_dir}/results-{args.test_language}-{args.dataset}-{args.num_train_epochs}epochs-{args.top_k}mvp-{args.weight_decay}-{args.warmup_steps}-{args.hidden_layers}{con_dec}{sent}{aspect}{best_ckpt}.json", 'w',  encoding="utf-8"), indent=2, sort_keys=True)
    

    return scores

# initialization
args = init_args()
seed_everything(args.seed, workers=True)

# training process
if args.do_train:
    tokenizer = MT5Tokenizer.from_pretrained(args.model_name_or_path)

    data_path = f"{args.path}/{args.dataset}_Restaurants_Train_{args.language}.json" 
    # Get example from the train set
    dataset = GenSCLNatDataset(tokenizer=tokenizer, data_type="Train", data_path=data_path, top_k=args.top_k, 
                            max_len=args.max_seq_length, task=args.task, args=args)
    data_sample = dataset[0]

    # sanity check
    # show one sample to check the code and the expected output format are correct
    print(f"Here is an example (from the train set):")
    print('Input :', tokenizer.decode(data_sample['source_ids'], skip_special_tokens=True))
    print(data_sample['source_ids'])
    print('Output:', tokenizer.decode(data_sample['target_ids'], skip_special_tokens=True))
    print(data_sample['target_ids'])

    print("\n****** Conducting Training ******")
    print(f"\nSentiment Contrastive Learning: {args.do_contr_sent}")
    print(f"\nAspect Contrastive Learning: {args.do_contr_aspect}")

    # initialize the MT5 model
    tfm_model = MT5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    tfm_model.resize_token_embeddings(len(tokenizer))
    # initialize characteristic-specific representation models
    cont_model = LinearModel()
    as_model = LinearModel()
    cat_model = LinearModel()
    model = MT5FineTuner(args, tfm_model, tokenizer, cont_model, as_model, cat_model)

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    sent = "sent" if args.do_contr_sent else ""
    aspect = "aspect" if args.do_contr_aspect else ""

    local_ckpt_dir = f"/content/tmp_ckpts_{args.language}_{args.top_k}_{aspect}_{sent}"
    os.makedirs(local_ckpt_dir, exist_ok=True)

    best_last5_cb = BestOfLastNEpochs(
        dirpath=ckpt_dir,
        filename="best-last5",
        monitor="val_f1",
        mode="max",
        n=5,
    )

    checkpoint_callback = SafeModelCheckpoint(
        dirpath=local_ckpt_dir,
        filename="best-checkpoint",
        monitor="val_f1",
        mode="max",
    )

    callback_list = [LoggingCallback(), best_last5_cb, checkpoint_callback]

    if args.early_stopping > 0:
        callback_list.append(
            EarlyStopping(monitor="val_f1", mode="max", patience=args.early_stopping)
        )

    train_params = dict(
        default_root_dir=args.output_dir,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        precision="bf16-mixed",
        accelerator='gpu', 
        devices='auto',
        gradient_clip_val=1.0,
        max_epochs=args.num_train_epochs,
        deterministic=True,
        callbacks=callback_list,
    )
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)

    if args.early_stopping:
        ex_weights = torch.load(checkpoint_callback.best_model_path)['state_dict']
        model.load_state_dict(ex_weights)
    
    shutil.copytree(local_ckpt_dir, ckpt_dir, dirs_exist_ok=True)
    print("Copied best‐checkpoint.ckpt to", ckpt_dir)
        
    model.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, 'args.json'), 'w',  encoding="utf-8") as f:
        json.dump(args.__dict__, f, indent=2)

    print("Finish training and saving the model!")

# evaluation
if args.do_direct_eval:
    print("\n****** Conduct Evaluating with the last state ******")
    print()
    data_path = f"{args.path}/{args.dataset}_Restaurants_Test_{args.test_language}.json" 

    test_dataset = GenSCLNatDataset(tokenizer, data_path=data_path, data_type="Test", top_k=args.top_k, max_len=args.max_seq_length, task=args.task, args=args)
    test_loader = DataLoader(test_dataset, args.eval_batch_size, num_workers=8)

    # compute the performance scores
    evaluate(test_loader, model, test_dataset.sentence_strings, args)

if args.do_inference:
    print("\n****** Conduct inference on trained checkpoint ******")

    # initialize the MT5 model from previous checkpoint
    model_path = args.model_name_or_path
    print(f"Loading trained model from {model_path}")
    tokenizer = MT5Tokenizer.from_pretrained(Path(model_path), local_files_only=True)
    tfm_model = MT5ForConditionalGeneration.from_pretrained(Path(model_path), local_files_only=True)

    # representations are only used during loss calculation
    cont_model = LinearModel()
    as_model = LinearModel()
    cat_model = LinearModel()
    model = MT5FineTuner(args, tfm_model, tokenizer, cont_model, as_model, cat_model)

    data_path = f"{args.path}/{args.dataset}_Restaurants_Test_{args.test_language}.json"

    print()
    test_dataset = GenSCLNatDataset(tokenizer, data_path=data_path, data_type="Test", top_k=args.top_k, max_len=args.max_seq_length, task=args.task, args=args)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, num_workers=8)

    # compute the performance scores
    evaluate(test_loader, model, test_dataset.sentence_strings, args)

if args.do_inference_best_ckpt or args.do_inference_best_5:
    print("\n****** Conduct inference on best checkpoint ******") if args.do_inference_best_ckpt else print ("\n****** Conduct inference on best checkpoint in final 5 epochs ******")

    # Instantiate from the best checkpoint
    ckpt_dir = os.path.join(args.model_name_or_path, "checkpoints")
    best_ckpt_path = os.path.join(ckpt_dir, "best-checkpoint.ckpt") if args.do_inference_best_ckpt else os.path.join(ckpt_dir, "best-last5.ckpt")
    if not os.path.exists(best_ckpt_path):
        raise FileNotFoundError(f"Best checkpoint not found at {best_ckpt_path}")

    print(f"Loading best checkpoint from {best_ckpt_path}")

    tokenizer = MT5Tokenizer.from_pretrained(Path(args.model_name_or_path), local_files_only=True)
    tfm_model = MT5ForConditionalGeneration.from_pretrained(Path(args.model_name_or_path), local_files_only=True)

    # representations are only used during loss calculation
    cont_model = LinearModel()
    as_model = LinearModel()
    cat_model = LinearModel()
    model = MT5FineTuner(args, tfm_model, tokenizer, cont_model, as_model, cat_model)

    checkpoint = torch.load(best_ckpt_path)
    model.load_state_dict(checkpoint["state_dict"])

    data_path = f"{args.path}/{args.dataset}_Restaurants_Test_{args.test_language}.json"

    print()
    test_dataset = GenSCLNatDataset(tokenizer, data_path=data_path, data_type="Test", top_k=args.top_k, max_len=args.max_seq_length, task=args.task, args=args)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, num_workers=8)

    # compute the performance scores
    evaluate(test_loader, model, test_dataset.sentence_strings, args)