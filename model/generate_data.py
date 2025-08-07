"""
Code is largely derived from https://github.com/jpeper/GEN_SCL_NAT.git
"""
# Need to edit this part

import numpy as np

use_the_gpu = False

def ex_contains_implicit_aspect(triplets):
    return any([triplet.get("a") == 'NULL' for triplet in triplets])

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll))
    return results

def get_pos_vec_bert(triplet, sent):
    """
    returns binary vector indicating tokens relevant to current triplet
    1 indicates token is part of either this triplets aspect span
    0 indicates it is not
    """
    span = triplet[0] # aspect and opinion terms
    zeroes_vec = np.zeros(len(sent))
    
    span_list = span.split(" ")
    # find locations of explicit terms to focus on
    curr_indices = find_sub_list(span_list, sent)
    if curr_indices:
        first_result = curr_indices[0]
        for idx in range(first_result[0], first_result[1]):
            zeroes_vec[idx] = 1

    return zeroes_vec

def get_cl_mvp_tasd_data(sents, labels, task):
    """
    Generate output target with the cl_mvp_tasd format
    """

    # 'cl_mvp_tasd_wo_intra', 'cl_mvp_tasd_wo_nat_cats', 'cl_mvp_tasd_wo_sorting'
    def inner_fn(sent, label):
        # handles a single review
        # get aspects
        aspect_terms = set()
        for triplet in label:
            aspect_terms.add(triplet.get("a"))

        utt_str = " ".join(sent)
        indices = {}
        for term in aspect_terms:
            if term.lower() != 'null': 
                indices[term] = utt_str.find(term)
            else:
                indices[term] = len(utt_str) + 1

        if task == 'cl_mvp_tasd_wo_sorting':
            sorted_aspects = aspect_terms
        else:
            sorted_aspects = sorted(aspect_terms, key=indices.get)


        outputs = []
        seen_aspects = set()
        covered = set()

        # progress through the review aspect by aspect
        # implements the scan-based ordering proposed in the NAT component of our work
        for aspect in sorted_aspects:
            seen_aspects.add(aspect)
            for triplet in label:
                # if we can produce the triplet using the seen aspects, then generate the summary
                if triplet.get("a") in seen_aspects and tuple((triplet.get("a"), triplet.get("c"), triplet.get("p"))) not in covered:
                    if len(triplet) == 3:
                        at, ac, ap = triplet.get("a"), triplet.get("c"), triplet.get("p")

                    covered.add(tuple((triplet.get("a"), triplet.get("c"), triplet.get("p"))))

                    if at.lower() == 'null':  # for implicit aspect term
                        at = 'it'

                    revised_triplet = [at, ac, ap]

                    pos_vec = get_pos_vec_bert(revised_triplet, sent)

                    outputs.append((revised_triplet, pos_vec))

        sent_len = len(sent)
        if task == 'cl_mvp_tasd_wo_sorting':
            sorted_outputs = outputs
        else:
            sorted_outputs = sorted(outputs, key = lambda x: (max(loc for loc, val in enumerate(x[1]) if val  == 1) if 1 in x[1] else sent_len, x[0]))
        total_sent = []
        
        for idx, output in enumerate(sorted_outputs):
            total_sent.append(output[0])
            
            if task not in ['cl_mvp_tasd']:
                print(task)
                print('NOT SUPPORTED')
                import pdb
                pdb.set_trace()

        return sent.copy(), total_sent

    sents, outputs = list(zip(*[inner_fn(sents[idx], labels[idx]) for idx in range(len(sents))]))
   
    return sents, outputs

