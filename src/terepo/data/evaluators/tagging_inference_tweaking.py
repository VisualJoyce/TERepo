"""
Copyright (c) VisualJoyce.
Licensed under the MIT license.
"""
import logging
import os.path
import random
from abc import abstractmethod
from collections import Counter
from typing import List

import numpy as np
import torch
from tqdm import tqdm

import ChERRANT.compare_m2_for_evaluation as cherrant_compare_m2
from ChERRANT.parallel_to_m2 import main as cherrant_parallel_to_m2
from m2scorer import m2scorer_f1_score
from terepo.data.evaluators import register_evaluator
from terepo.data.evaluators.base import TERepoBaseEvaluator
from terepo.data.evaluators.tagging import convert_para_to_edits
from terepo.data.loaders import move_to_cuda

logger = logging.getLogger(__name__)


class TERepoInferenceTweakingBaseEvaluator(TERepoBaseEvaluator):

    @abstractmethod
    def f1_score(self, src_texts,
                 pred_texts,
                 trg_texts, output_dir):
        raise NotImplementedError

    def logits_to_labels_thresholded(self, p_labels, err_prob):
        ret = []
        for min_error_probability in np.arange(0.3, 0.5, 0.01):
            for confidence_bias in np.arange(0, 0.2, 0.05):
                label_probs = p_labels.detach().clone()
                label_probs[:, self.feature_extractor.labels_keep_token_id] += confidence_bias
                if err_prob < min_error_probability:
                    label_ids = [self.feature_extractor.labels_keep_token_id] * label_probs.size(0)
                else:
                    label_ids = []
                    for i, lp in enumerate(label_probs):
                        prob, idx = torch.max(lp, dim=0)
                        if prob < min_error_probability:
                            label_ids.append(self.feature_extractor.labels_keep_token_id)
                        else:
                            label_ids.append(idx.item())
                labels = self.feature_extractor.convert_ids_to_labels_list([[x] for x in label_ids])
                ret.append((min_error_probability, confidence_bias, labels))
        return ret

    def parse_logits(self, sources, input_words, output_words, golden_labels, input_mask, logits_labels, logits_d):
        logits_labels.masked_fill_(input_mask.unsqueeze(2) == 0, -1e-9)
        logits_d.masked_fill_(input_mask.unsqueeze(2) == 0, -1e-9)
        class_probabilities_labels = torch.softmax(logits_labels, dim=-1)
        class_probabilities_d = torch.softmax(logits_d, dim=-1)
        error_probs = class_probabilities_d[:, :, self.feature_extractor.dtags_incorrect_id]
        incorrect_prob = torch.max(error_probs, dim=-1)[0]
        for source, s_tokens, t_tokens, golden_label_ids, p_labels, err_prob in zip(sources, input_words, output_words,
                                                                                    golden_labels,
                                                                                    class_probabilities_labels,
                                                                                    incorrect_prob):
            golden_labels_list = self.feature_extractor.convert_ids_to_labels_list(
                [[x] for x in golden_label_ids.detach().tolist()])
            yield source, s_tokens, t_tokens, golden_labels_list, self.logits_to_labels_thresholded(p_labels, err_prob)

    def logits_list_to_labels_thresholded(self, p_labels_list, err_prob):
        ret = []
        for min_error_probability in np.arange(0.3, 0.5, 0.01):
            for confidence_bias in np.arange(0, 0.2, 0.05):
                label_ids = [[] for _ in range(p_labels_list[0].size(0))]
                for i_step, p_labels in enumerate(p_labels_list):
                    label_probs = p_labels.detach().clone()
                    label_probs[:, self.feature_extractor.labels_keep_token_id] += confidence_bias
                    if err_prob < min_error_probability and i_step == 0:
                        label_ids[0] = [self.feature_extractor.labels_keep_token_id] * label_probs.size(0)
                    else:
                        for i, lp in enumerate(label_probs):
                            prob, idx = torch.max(lp, dim=0)
                            if prob < min_error_probability:
                                label_ids[i].append(self.feature_extractor.labels_keep_token_id)
                            else:
                                label_ids[i].append(idx.item())
                labels = self.feature_extractor.convert_ids_to_labels_list([[x] for x in label_ids])
                ret.append((min_error_probability, confidence_bias, labels))
        return ret

    def forwards(self, model, loader):
        for step, batch in enumerate(tqdm(loader)):
            sources, input_words, output_words = batch.pop('meta')
            batch = move_to_cuda(batch, device=self.training_args.device)
            outputs = model(**batch, return_dict=True)

            for item in self.parse_logits(sources, input_words, output_words,
                                          batch['labels'],
                                          batch['original_mask'],
                                          outputs.logits_labels,
                                          outputs.logits_d):
                yield item

    def __call__(self, model, loader, output_dir):
        src_texts, pred_texts_dict, trg_texts = [], {}, []
        errors = []
        for source, s_tokens, t_tokens, golden_labels, labels_list_all in self.forwards(model, loader):
            s_sent = self.feature_extractor.convert_tokens_to_string(s_tokens[1:], self.tokenizer)
            t_sent = self.feature_extractor.convert_tokens_to_string(t_tokens[1:], self.tokenizer)
            src_texts.append(source)
            trg_texts.append(t_sent)

            for min_error_probability, confidence_bias, labels_list in labels_list_all:
                p = self.feature_extractor.convert_labels_list_to_sentence(s_tokens, labels_list)
                p_sent = self.feature_extractor.convert_tokens_to_string(p[1:], self.tokenizer)

                pred_texts_dict.setdefault((min_error_probability, confidence_bias), [])
                pred_texts_dict[(min_error_probability, confidence_bias)].append(p_sent)
                if t_sent != p_sent:
                    errors.append((s_sent, t_sent, golden_labels, labels_list, p_sent))

        for s_tokens, t_tokens, g_labels, p_labels, p in random.choices(errors, k=min(len(errors), 10)):
            print('---------------------------------')
            print(f'  source: {s_tokens}')
            print(f'  target: {t_tokens}')
            print(f'g_labels: {g_labels}')
            print(f'p_labels: {p_labels}')
            print(f' predict: {p}')

        metrics = {}
        for k, pred_texts in pred_texts_dict.items():
            metrics[k] = self.f1_score(src_texts, pred_texts, trg_texts, output_dir)

        return metrics


@register_evaluator("tagging_inference_tweaking", "cherrant")
class TERepoInferenceTweakingChErrantEvaluator(TERepoInferenceTweakingBaseEvaluator):
    def compare_m2_for_evaluation(self, args):
        # Parse command line args
        # Open hypothesis and reference m2 files and split into chunks
        hyp_m2 = open(args.hyp).read().strip().split("\n\n")[
                 args.start:args.end] if args.start is not None and args.end is not None else open(
            args.hyp).read().strip().split("\n\n")
        ref_m2 = open(args.ref).read().strip().split("\n\n")[
                 args.start:args.end] if args.start is not None and args.end is not None else open(
            args.ref).read().strip().split("\n\n")
        # Make sure they have the same number of sentences
        assert len(hyp_m2) == len(ref_m2), print(len(hyp_m2), len(ref_m2))

        # Store global corpus level best counts here
        best_dict = Counter({"tp": 0, "fp": 0, "fn": 0})
        best_cats = {}
        # Process each sentence
        sents = zip(hyp_m2, ref_m2)
        for sent_id, sent in enumerate(sents):
            # Simplify the edits into lists of lists
            # if "A1" in sent[0] or "A1" in sent[1] or sent_id in sent_id_cons:
            #     sent_id_cons.append(sent_id)
            src = sent[0].split("\n")[0]
            hyp_edits = cherrant_compare_m2.simplify_edits(sent[0], args.max_answer_num)
            ref_edits = cherrant_compare_m2.simplify_edits(sent[1], args.max_answer_num)
            # Process the edits for detection/correction based on args
            hyp_dict = cherrant_compare_m2.process_edits(hyp_edits, args)
            ref_dict = cherrant_compare_m2.process_edits(ref_edits, args)
            if args.reference_num is None or len(ref_dict.keys()) == args.reference_num:
                # Evaluate edits and get best TP, FP, FN hyp+ref combo.
                count_dict, cat_dict = cherrant_compare_m2.evaluate_edits(src,
                                                                          hyp_dict, ref_dict, best_dict, sent_id, args)
                # Merge these dicts with best_dict and best_cats
                best_dict += Counter(count_dict)
                best_cats = cherrant_compare_m2.merge_dict(best_cats, cat_dict)

        # Prepare output title.
        if args.dt:
            title = " Token-Based Detection "
        elif args.ds:
            title = " Span-Based Detection "
        elif args.cse:
            title = " Span-Based Correction + Classification "
        else:
            title = " Span-Based Correction "

        # Category Scores
        if args.cat:
            best_cats = cherrant_compare_m2.processCategories(best_cats, args.cat)
            print("")
            print('{:=^66}'.format(title))
            print("Category".ljust(14), "TP".ljust(8), "FP".ljust(8), "FN".ljust(8),
                  "P".ljust(8), "R".ljust(8), "F" + str(args.beta))
            for cat, cnts in sorted(best_cats.items()):
                cat_p, cat_r, cat_f = cherrant_compare_m2.computeFScore(cnts[0], cnts[1], cnts[2], args.beta)
                print(cat.ljust(14), str(cnts[0]).ljust(8), str(cnts[1]).ljust(8),
                      str(cnts[2]).ljust(8), str(cat_p).ljust(8), str(cat_r).ljust(8), cat_f)

        # Print the overall results.
        p, r, f = cherrant_compare_m2.computeFScore(best_dict["tp"], best_dict["fp"], best_dict["fn"], args.beta)
        print("")
        print('{:=^46}'.format(title))
        print("\t".join(["TP", "FP", "FN", "Prec", "Rec", "F" + str(args.beta)]))
        print("\t".join(map(str, [best_dict["tp"], best_dict["fp"],
                                  best_dict["fn"]] + list((p, r, f)))))
        print('{:=^46}'.format(""))
        print("")
        return {
            "precision": p,
            "recall": r,
            "f1": f
        }

    def f1_score(self, src_texts,
                 pred_texts,
                 trg_texts, output_dir):
        """
        1. 将标准答案平行文件通过`parallel_to_m2.py`转换成M2格式的编辑文件`gold.m2`（仅首次评估需要，之后可以复用）；
        2. 将预测答案平行文件通过`parallel_to_m2.py`转换成M2格式的编辑文件`hyp.m2`；
        3. 使用`compare_m2_for_evaluation.py`对比`hyp.m2`和`gold.m2`，得到最终的评价指标。
        """

        class ArgumentsStub:
            batch_size = 128
            device = 0
            worker_num = 16
            granularity = 'char'
            merge = False
            multi_cheapest_strategy = 'all'
            segmented = False
            no_simplified = False

            def __init__(self, input_file):
                self.file = input_file
                self.output = f'{input_file}.{self.granularity}'

        eval_gold_file = self.data_args.eval_gold_file
        args_gold = ArgumentsStub(eval_gold_file)
        if not os.path.isfile(args_gold.output):
            cherrant_parallel_to_m2(args_gold)

        src_texts_uniq = {}
        for s, p, t in zip(src_texts, pred_texts, trg_texts):
            if s not in src_texts_uniq:
                src_texts_uniq[s] = p, [t]
            else:
                src_texts_uniq[s][1].append(t)

        eval_prediction_file = os.path.join(output_dir, 'prediction.txt')
        with open(eval_prediction_file, 'w') as fw:
            with open(eval_gold_file) as f:
                for l in f:
                    idx, source = l.split('\t')[:2]
                    fw.write(f'{idx}\t{source}\t{src_texts_uniq[source][0]}\n')
        args_eval = ArgumentsStub(eval_prediction_file)
        cherrant_parallel_to_m2(args_eval)

        class EvaluationArgumentsStub:
            start = None
            end = None
            max_answer_num = None
            reference_num = None
            beta = 0.5
            verbose = False
            no_simplified = False
            dt = False
            ds = False
            cs = False
            cse = False
            single = False
            multi = False
            multi_hyp_avg = False
            multi_hyp_max = False
            filt = []
            cat = [1, 2, 3]

            def __init__(self, hyp, ref):
                self.hyp = hyp
                self.ref = ref

        evaluation_arguments = EvaluationArgumentsStub(args_eval.output, args_gold.output)
        return self.compare_m2_for_evaluation(evaluation_arguments)


@register_evaluator("tagging_inference_tweaking", "errant")
class TERepoInferenceTweakingErrantEvaluator(TERepoInferenceTweakingBaseEvaluator):

    def f1_score(self, src_texts,
                 pred_texts,
                 trg_texts, output_dir):
        max_unchanged_words = 2
        beta = 0.5
        ignore_whitespace_casing = False
        verbose = False
        very_verbose = False

        src_texts_uniq = {}
        for s, p, t in zip(src_texts, pred_texts, trg_texts):
            if s not in src_texts_uniq:
                src_texts_uniq[s] = p, [t]
            else:
                src_texts_uniq[s][1].append(t)

        source_sentences, system_sentences, target_sentences = [], [], []
        for s, (p, t) in src_texts_uniq.items():
            source_sentences.append(s)
            system_sentences.append(p)
            target_sentences.append(t)

        # load source sentences and gold edits
        gold_edits = convert_para_to_edits(source_sentences, target_sentences, lang='en')
        p, r, f1 = m2scorer_f1_score(system_sentences, source_sentences, gold_edits,
                                     max_unchanged_words, beta, ignore_whitespace_casing, verbose,
                                     very_verbose)
        logger.info("Precision   : %.4f" % p)
        logger.info("Recall      : %.4f" % r)
        return {
            'precision': p,
            'recall': r,
            'f1': f1
        }
