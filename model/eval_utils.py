"""
Code is largely derived from https://github.com/jpeper/GEN_SCL_NAT.git and 
https://github.com/ZubinGou/multi-view-prompting.git
"""

# -*- coding: utf-8 -*-
import numpy as np

def extract_spans_para(seq, seq_type):
    triplets = []
    sents = [s.strip() for s in seq.split('[SSEP]')]
    for s in sents:
        try:
            tok_list = ["[C]", "[P]", "[A]"]

            for tok in tok_list:
                if tok not in s:
                    s += " {} null".format(tok)
            index_ac = s.index("[C]")
            index_sp = s.index("[P]")
            index_at = s.index("[A]")

            combined_list = [index_ac, index_sp, index_at]
            arg_index_list = list(np.argsort(combined_list))

            result = []
            for i in range(len(combined_list)):
                start = combined_list[i] + 3
                sort_index = arg_index_list.index(i)
                if sort_index < 2:
                    next_ = arg_index_list[sort_index + 1]
                    re = s[start:combined_list[next_]]
                else:
                    re = s[start:]
                result.append(re.strip())

            ac, sp, at = result

        except ValueError:
            try:
                print(f'In {seq_type} seq, cannot decode: {s}')
                pass
            except UnicodeEncodeError:
                print(f'In {seq_type} seq, a string cannot be decoded')
                pass
            ac, at, sp = '', '', ''

        triplets.append((ac, at, sp))

    return triplets
    
def compute_f1_scores(pred_pt, gold_pt, silent=True):
    """
    Function to compute F1 scores with pred and gold triplets
    The input needs to be already processed
    """
    # number of true postive, gold standard, predictions
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(set(gold_pt[i]))
        n_pred += len(set(pred_pt[i]))

        for t in set(pred_pt[i]):
            if t in gold_pt[i]:
                n_tp += 1

    

    if not silent:
        print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}")
    
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    if recall > 1.0:
        import pdb
        pdb.set_trace()
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1, 'num_gold_spans': n_gold, 'num_predicted_spans': n_pred, 'hit': n_tp}

    return scores


def compute_scores(pred_seqs, gold_seqs, silent=True):
    """
    Compute model performance
    """
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_preds = [], []

    for i in range(num_samples):
        gold_list = extract_spans_para(seq=gold_seqs[i],seq_type="gold")
        pred_list = extract_spans_para(seq=pred_seqs[i],seq_type="pred")

        all_labels.append(gold_list)
        all_preds.append(pred_list)

    if not silent:
        print("\nResults:")
        scores = compute_f1_scores(all_preds, all_labels, silent)
        print(scores)
    else:
        scores = compute_f1_scores(all_preds, all_labels, silent)

    return scores, all_labels, all_preds
