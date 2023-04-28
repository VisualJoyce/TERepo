"""
Copyright (c) VisualJoyce.
Licensed under the MIT license.
"""
import io
import logging
import os.path
import random
from abc import abstractmethod
from collections import Counter, OrderedDict
from typing import List

import errant
import torch
from tqdm import tqdm

from errant.commands.compare_m2 import simplify_edits, process_edits, merge_dict, evaluate_edits, print_results, \
    computeFScore
import ChERRANT.compare_m2_for_evaluation as cherrant_compare_m2
from ChERRANT.parallel_to_m2 import main as cherrant_parallel_to_m2
from m2scorer import m2scorer_f1_score
from terepo.data.evaluators import register_evaluator
from terepo.data.evaluators.base import TERepoBaseEvaluator
from terepo.data.loaders import move_to_cuda

logger = logging.getLogger(__name__)


class TERepoTaggingBaseEvaluator(TERepoBaseEvaluator):

    @abstractmethod
    def f1_score(self, src_texts,
                 pred_texts,
                 trg_texts, output_dir):
        raise NotImplementedError

    def parse_logits(self, sources, input_words, output_words, golden_labels, logits_labels):
        for source, s_tokens, t_tokens, golden_label_ids, l_labels in zip(sources, input_words, output_words,
                                                                          golden_labels,
                                                                          logits_labels):
            label_ids = []
            for i, token in enumerate(s_tokens):
                logits_label = l_labels[i]
                idx = torch.argmax(logits_label).item()
                label_ids.append([idx])
            yield source, s_tokens, t_tokens, self.feature_extractor.convert_ids_to_labels_list(
                [[x] for x in golden_label_ids.detach().tolist()]), self.feature_extractor.convert_ids_to_labels_list(
                label_ids)

    def parse_logits_list(self, sources, input_words, output_words, golden_labels, logits_labels_list,
                          logits_tags_list):
        for i_ex, (source, s_tokens, t_tokens, golden_label_ids) in enumerate(
                zip(sources, input_words, output_words, golden_labels)):
            label_ids = [[] for _ in s_tokens]
            l_labels_list = [item[i_ex] for item in logits_labels_list]
            l_tags_list = [item[i_ex] for item in logits_tags_list]
            for i_step, (logits_labels, logits_tags) in enumerate(zip(l_labels_list, l_tags_list)):
                for i, token in enumerate(s_tokens):
                    logits_label = logits_labels[i]
                    idx = torch.argmax(logits_label).item()

                    # logits_tag = logits_tags[i]
                    # d_idx = torch.argmax(logits_tag).item()
                    label_ids[i].append(idx)

                    # if d_idx != self.tokenizer.dtags_correct_id:
                    #     label_ids[i].append(idx)
                    # else:
                    #     label_ids[i].append(self.tokenizer.labels_keep_token_id)
            yield source, s_tokens, t_tokens, self.feature_extractor.convert_ids_to_labels_list(
                golden_label_ids.detach().tolist()), self.feature_extractor.convert_ids_to_labels_list(label_ids)

    def forwards(self, model, loader):
        for step, batch in enumerate(tqdm(loader)):
            sources, input_words, output_words = batch.pop('meta')
            batch = move_to_cuda(batch, device=self.training_args.device)
            outputs = model(**batch, return_dict=True)

            # class_probabilities_labels = torch.softmax(outputs.logits_labels, dim=-1)
            # class_probabilities_d = torch.softmax(outputs.logits_d, dim=-1)
            # error_probs = class_probabilities_d[:, :, self.tokenizer.dtags_incorrect_id] * mask
            # incorrect_prob = torch.max(error_probs, dim=-1)[0]
            if isinstance(outputs.logits_labels, List):
                for item in self.parse_logits_list(sources, input_words, output_words, batch['labels'],
                                                   outputs.logits_labels,
                                                   outputs.logits_d):
                    yield item
            else:
                for item in self.parse_logits(sources, input_words, output_words, batch['labels'],
                                              outputs.logits_labels):
                    yield item

    def __call__(self, model, loader, output_dir):
        src_texts, pred_texts, trg_texts = [], [], []
        errors = []
        for source, s_tokens, t_tokens, golden_labels, labels_list in self.forwards(model, loader):
            s_sent = self.feature_extractor.convert_tokens_to_string(s_tokens[1:], self.tokenizer)
            t_sent = self.feature_extractor.convert_tokens_to_string(t_tokens[1:], self.tokenizer)

            p = self.feature_extractor.convert_labels_list_to_sentence(s_tokens, labels_list)
            p_sent = self.feature_extractor.convert_tokens_to_string(p[1:], self.tokenizer)

            src_texts.append(source)
            trg_texts.append(t_sent)
            pred_texts.append(p_sent)
            if t_sent != p_sent:
                errors.append((s_sent, t_sent, golden_labels, labels_list, p_sent))

        for s_tokens, t_tokens, g_labels, p_labels, p in random.choices(errors, k=min(len(errors), 10)):
            print('---------------------------------')
            print(f'  source: {s_tokens}')
            print(f'  target: {t_tokens}')
            print(f'g_labels: {g_labels}')
            print(f'p_labels: {p_labels}')
            print(f' predict: {p}')
        return self.f1_score(src_texts, pred_texts, trg_texts, output_dir)


@register_evaluator("tagging", "cherrant")
class TERepoTaggingChErrantEvaluator(TERepoTaggingBaseEvaluator):
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


def noop_edit(id=0):
    return "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||" + str(id)


def convert_para_to_m2(src_texts, trg_texts, tokenize=False, lev=False, merge="rules", lang=None):
    """

    :param lang:
    :param src_texts:
    :param trg_texts:
    :param tokenize:
    :param lev:
    :param merge: help="Choose a merging strategy for automatic alignment.\n"
            "rules: Use a rule-based merging strategy (default)\n"
            "all-split: Merge nothing: MSSDI -> M, S, S, D, I\n"
            "all-merge: Merge adjacent non-matches: MSSDI -> M, SSDI\n"
            "all-equal: Merge adjacent same-type non-matches: MSSDI -> M, SS, D, I",
        choices=["rules", "all-split", "all-merge", "all-equal"],
    :return:
    """
    out_m2 = io.StringIO()
    annotator = errant.load(lang)

    # Process each line of all input files
    for orig, cors in zip(src_texts, trg_texts):
        # Skip the line if orig is empty
        if not orig: continue
        # Parse orig with spacy
        orig = annotator.parse(orig, tokenize)
        # Write orig to the output m2 file
        out_m2.write(" ".join(["S"] + [token.text for token in orig]) + "\n")
        assert isinstance(cors, list)
        # Loop through the corrected texts
        for cor_id, cor in enumerate(cors):
            cor = cor.strip()
            # If the texts are the same, write a noop edit
            if orig.text.strip() == cor:
                out_m2.write(noop_edit(cor_id) + "\n")
            # Otherwise, do extra processing
            else:
                # Parse cor with spacy
                cor = annotator.parse(cor, tokenize)
                # Align the texts and extract and classify the edits
                edits = annotator.annotate(orig, cor, lev, merge)
                # Loop through the edits
                for edit in edits:
                    # Write the edit to the output m2 file
                    out_m2.write(edit.to_m2(cor_id) + "\n")
        # Write a newline when we have processed all corrections for each line
        out_m2.write("\n")
    return out_m2


def paragraphs(lines, is_separator=lambda x: x == '\n', joiner=''.join):
    paragraph = []
    for line in lines:
        if is_separator(line):
            if paragraph:
                yield joiner(paragraph)
                paragraph = []
        else:
            paragraph.append(line)
    if paragraph:
        yield joiner(paragraph)


def convert_para_to_edits(src_texts, trg_texts, lang):
    gold_edits = []
    fd = convert_para_to_m2(src_texts, trg_texts, tokenize=True, lang=lang)
    for item in paragraphs(fd.getvalue().splitlines(True)):
        item = item.splitlines(False)
        sentence = [line[2:].strip() for line in item if line.startswith('S ')]
        assert sentence != []
        annotations = {}
        for line in item[1:]:
            if line.startswith('I ') or line.startswith('S '):
                continue
            assert line.startswith('A ')
            line = line[2:]
            fields = line.split('|||')
            start_offset = int(fields[0].split()[0])
            end_offset = int(fields[0].split()[1])
            etype = fields[1]
            if etype == 'noop':
                start_offset = -1
                end_offset = -1
            corrections = [c.strip() if c != '-NONE-' else '' for c in fields[2].split('||')]
            # NOTE: start and end are *token* offsets
            original = ' '.join(' '.join(sentence).split()[start_offset:end_offset])
            annotator = int(fields[5])
            if annotator not in list(annotations.keys()):
                annotations[annotator] = []
            annotations[annotator].append((start_offset, end_offset, original, corrections))
        tok_offset = 0
        for this_sentence in sentence:
            tok_offset += len(this_sentence.split())
            this_edits = {}
            for annotator, annotation in annotations.items():
                this_edits[annotator] = [edit for edit in annotation if
                                         edit[0] <= tok_offset and edit[1] <= tok_offset and edit[0] >= 0 and edit[
                                             1] >= 0]
            if len(this_edits) == 0:
                this_edits[0] = []
            gold_edits.append(this_edits)
    return gold_edits


@register_evaluator("tagging", "m2scorer")
class TERepoM2ScorerEvaluator(TERepoTaggingBaseEvaluator):

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


@register_evaluator("tagging", "errant")
class TERepoErrantEvaluator(TERepoTaggingBaseEvaluator):

    def parallel_to_m2(self, source_sentences, system_sentences, output_dir):
        # Input: A coder id
        # Output: A noop edit; i.e. text contains no edits
        def noop_edit(id=0):
            return "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||" + str(id)

        class ArgumentsStub:
            out = os.path.join(output_dir, 'prediction.m2')
            tok = False
            lev = False
            merge = 'rules'

        args = ArgumentsStub()

        print("Loading resources...")
        # Load Errant
        annotator = errant.load("en")

        print("Processing parallel files...")
        # Process an arbitrary number of files line by line simultaneously. Python 3.3+
        # See https://tinyurl.com/y4cj4gth . Also opens the output m2 file.
        with open(args.out, "w") as out_m2:
            # Process each line of all input files
            for orig, cors in tqdm(zip(source_sentences, system_sentences), total=len(source_sentences)):
                # Skip the line if orig is empty
                if not orig: continue
                # Parse orig with spacy
                orig = annotator.parse(orig, args.tok)
                # Write orig to the output m2 file
                out_m2.write(" ".join(["S"] + [token.text for token in orig]) + "\n")
                # Loop through the corrected texts
                for cor_id, cor in enumerate(cors):
                    cor = cor.strip()
                    # If the texts are the same, write a noop edit
                    if orig.text.strip() == cor:
                        out_m2.write(noop_edit(cor_id) + "\n")
                    # Otherwise, do extra processing
                    else:
                        # Parse cor with spacy
                        cor = annotator.parse(cor, args.tok)
                        # Align the texts and extract and classify the edits
                        edits = annotator.annotate(orig, cor, args.lev, args.merge)
                        # Loop through the edits
                        for edit in edits:
                            # Write the edit to the output m2 file
                            out_m2.write(edit.to_m2(cor_id) + "\n")
                # Write a newline when we have processed all corrections for each line
                out_m2.write("\n")

    def compare_m2(self, output_dir):
        class ArgumentsStub:
            hyp = os.path.join(output_dir, 'prediction.m2')
            ref = self.data_args.eval_gold_file
            beta = 0.5
            verbose = False
            dt = False
            ds = False
            cs = False
            cse = False
            single = False
            multi = False
            filt = []
            cat = None

        # Parse command line args
        args = ArgumentsStub()
        # Open hypothesis and reference m2 files and split into chunks
        hyp_m2 = open(args.hyp).read().strip().split("\n\n")
        ref_m2 = open(args.ref).read().strip().split("\n\n")
        # Make sure they have the same number of sentences
        assert len(hyp_m2) == len(ref_m2)

        # Store global corpus level best counts here
        best_dict = Counter({"tp": 0, "fp": 0, "fn": 0})
        best_cats = {}
        # Process each sentence
        sents = zip(hyp_m2, ref_m2)
        for sent_id, sent in enumerate(sents):
            # Simplify the edits into lists of lists
            hyp_edits = simplify_edits(sent[0])
            ref_edits = simplify_edits(sent[1])
            # Process the edits for detection/correction based on args
            hyp_dict = process_edits(hyp_edits, args)
            ref_dict = process_edits(ref_edits, args)
            # original sentence for logging
            original_sentence = sent[0][2:].split("\nA")[0]
            # Evaluate edits and get best TP, FP, FN hyp+ref combo.
            count_dict, cat_dict = evaluate_edits(
                hyp_dict, ref_dict, best_dict, sent_id, original_sentence, args)
            # Merge these dicts with best_dict and best_cats
            best_dict += Counter(count_dict)
            best_cats = merge_dict(best_cats, cat_dict)
        # Print results
        print_results(best_dict, best_cats, args)
        return computeFScore(best_dict["tp"], best_dict["fp"], best_dict["fn"], args.beta)

    def f1_score(self, src_texts,
                 pred_texts,
                 trg_texts, output_dir):
        src_texts_uniq = OrderedDict()
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

        self.parallel_to_m2(source_sentences, system_sentences, output_dir)
        p, r, f1 = self.compare_m2(output_dir)
        return {
            'precision': p,
            'recall': r,
            'f1': f1
        }
