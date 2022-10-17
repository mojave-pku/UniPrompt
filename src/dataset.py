import os
import copy
import logging
import torch
import numpy as np
import time
from tqdm import tqdm
from filelock import FileLock
import json
import itertools
import random
import transformers
from src.processors import num_labels_mapping, output_modes_mapping, compute_metrics_mapping, median_mapping, TextClassificationProcessor
from transformers.data.processors.utils import InputFeatures
from transformers import DataProcessor, InputExample
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Union
from copy import deepcopy
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class OurInputFeatures(InputFeatures):
    input_ids: List[int]
    prompt_input_ids: List[int] = None
    attention_mask: Optional[List[int]] = None
    prompt_attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: List[int] = None
    mask_pos: Optional[List[int]] = None 
    label_word_list: Optional[List[int]] = None
    mask_lm_label: Optional[int] = None
    is_mlm_case: Optional[List[bool]] = None
    is_for_soft_label: Optional[bool] = None

    def to_json_string(self):
        return json.dumps(dataclasses.asdict(self)) + "\n"

def input_example_to_string(example, sep_token): 
    if example.text_b is None:
        return example.text_a
    else:
        return example.text_a + ' ' + sep_token + ' ' + example.text_b

def input_example_to_tuple(example): 
    if example.text_b is None:
        if pd.isna(example.text_a) or example.text_a is None:
            return ['']
            logger.warn("Empty input")
        else:
            return [example.text_a]
    else:
        return [example.text_a, example.text_b]

def paddings(input_ids, max_length, tokenizer, attention_mask, labels, token_type_ids=None):
    while len(input_ids) < max_length:
        input_ids.append(tokenizer.pad_token_id)
        attention_mask.append(0)
        labels.append(-100)
        if token_type_ids is not None:
            token_type_ids.append(0)
    if token_type_ids is not None:
        return input_ids, attention_mask, labels, token_type_ids
    else:
        return input_ids, attention_mask, labels

def tokenize_multipart_input(
    input_text_list, 
    max_length, 
    tokenizer, 
    task_name=None, 
    prompt=False, 
    template=None,
    label_word_list=None, 
    first_sent_limit=None,
    other_sent_limit=None,
    gpt3=False,
    truncate_head=False,
    support_labels=None,
    label=None,
    need_padding=True
):
    def enc(text):
        return tokenizer.encode(text, add_special_tokens=False)

    input_ids = []
    attention_mask = []
    token_type_ids = [] 
    mask_pos = None 

    if prompt:
        assert template is not None
        special_token_mapping = {
            'cls': tokenizer.cls_token_id, 'mask': tokenizer.mask_token_id, 'sep': tokenizer.sep_token_id, 'sep+': tokenizer.sep_token_id, 
        }
        template_list = template.split('*') 
        segment_id = 0 

        for part_id, part in enumerate(template_list):
            new_tokens = []
            segment_plus_1_flag = False
            if part in special_token_mapping:
                if part == 'cls' and 'T5' in type(tokenizer).__name__:
                    continue
                new_tokens.append(special_token_mapping[part])
                if part == 'sep+':
                    segment_plus_1_flag = True
            elif part[:6] == 'label_':
                label_id = int(part.split('_')[1])
                label_word = label_word_list[label_id]
                new_tokens.append(label_word)
            elif part[:7] == 'labelx_':
                instance_id = int(part.split('_')[1])
                label_id = support_labels[instance_id]
                label_word = label_word_list[label_id]
                new_tokens.append(label_word)
            elif part[:5] == 'sent_':
                sent_id = int(part.split('_')[1])
                new_tokens += enc(input_text_list[sent_id]) 
            elif part[:6] == '+sent_':
                sent_id = int(part.split('_')[1])
                new_tokens += enc(' ' + input_text_list[sent_id])
            elif part[:6] == 'sent-_':
                sent_id = int(part.split('_')[1])
                new_tokens += enc(input_text_list[sent_id][:-1])
            elif part[:6] == 'sentl_':
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_tokens += enc(text)
            elif part[:7] == '+sentl_':
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_tokens += enc(' ' + text)
            elif part[:7] == 'sentl-_':
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_tokens += enc(text[:-1])
            elif part[:6] == 'sentu_':
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].upper() + text[1:]
                new_tokens += enc(text)
            elif part[:7] == '+sentu_':
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].upper() + text[1:]
                new_tokens += enc(' ' + text)
            else:
                part = part.replace('_', ' ') 
                if len(part) == 1:
                    new_tokens.append(tokenizer._convert_token_to_id(part))
                else:
                    new_tokens += enc(part)

            if part[:4] == 'sent' or part[1:5] == 'sent':
                sent_id = int(part.split('_')[1])
                if sent_id == 0:
                    if first_sent_limit is not None:
                        new_tokens = new_tokens[:first_sent_limit]
                else:
                    if other_sent_limit is not None:
                        new_tokens = new_tokens[:other_sent_limit]

            input_ids += new_tokens
            attention_mask += [1 for i in range(len(new_tokens))]
            token_type_ids += [segment_id for i in range(len(new_tokens))]

            if segment_plus_1_flag:
                segment_id += 1
    else:
        input_ids = [tokenizer.cls_token_id]
        attention_mask = [1]
        token_type_ids = [0]

        for sent_id, input_text in enumerate(input_text_list):
            if input_text is None:
                continue
            if pd.isna(input_text) or input_text is None:
                input_text = ''
            input_tokens = enc(input_text) + [tokenizer.sep_token_id]
            input_ids += input_tokens
            attention_mask += [1 for i in range(len(input_tokens))]
            token_type_ids += [sent_id for i in range(len(input_tokens))]

        if 'T5' in type(tokenizer).__name__: 
            input_ids = input_ids[1:]
            attention_mask = attention_mask[1:]
            token_type_ids = token_type_ids[1:]

    if need_padding:
        if first_sent_limit is not None and len(input_ids) > max_length:
            logger.warn("Input exceeds max_length limit: {}".format(tokenizer.decode(input_ids)))

        while len(input_ids) < max_length:
            input_ids.append(tokenizer.pad_token_id)
            attention_mask.append(0)
            token_type_ids.append(0)

    if len(input_ids) > max_length:
        original_input_ids = input_ids
        if truncate_head:
            input_ids = input_ids[-max_length:]
            attention_mask = attention_mask[-max_length:]
            token_type_ids = token_type_ids[-max_length:]
        else:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            token_type_ids = token_type_ids[:max_length]

    if prompt:
        mask_pos = [input_ids.index(tokenizer.mask_token_id)]
        assert mask_pos[0] < max_length
    result = {'input_ids': input_ids, 'attention_mask': attention_mask}
    if 'BERT' in type(tokenizer).__name__:
        result['token_type_ids'] = token_type_ids

    if prompt:
        result['mask_pos'] = mask_pos
        result['label_word_list'] = label_word_list
        mlm_labels = [-100 for _ in range(len(input_ids))]
        mlm_labels[mask_pos[0]] = label_word_list[label]
        result['label'] = mlm_labels
    else:
        result['label'] = label

    return result



class FewShotDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, cache_dir=None, mode="train", use_demo=False, only_fetch_label=None):
        self.args = args
        self.task_name = args.task_name
        self.tokenizer = tokenizer
        self.mode = mode

        self.only_fetch_label = only_fetch_label

        if mode == "train" or mode == "dev":
            self.mapping = args.src_mapping
            self.template = args.src_template
            self.data_dir = args.src_data_dir
        elif mode == "test":
            self.mapping = args.dst_mapping
            self.template = args.dst_template
            self.data_dir = args.dst_data_dir

        if self.args.enable_two_tower_model:
            inputs = tokenize_multipart_input(
                input_text_list=[],
                max_length=self.args.max_seq_length,
                tokenizer=self.tokenizer,
                task_name=self.args.task_name,
                prompt=True,
                template=self.template,
                label_word_list=self.mapping,
                first_sent_limit=self.args.first_sent_limit,
                other_sent_limit=self.args.other_sent_limit,
                label=0
            )
            self.prompt_length = sum(inputs['attention_mask'])
            self.context_length = self.args.max_seq_length - self.prompt_length

        self.processor = TextClassificationProcessor(args, self.only_fetch_label)

        self.use_demo = use_demo
        if self.use_demo:
            logger.info("Use demonstrations")
        assert mode in ["train", "dev", "test"]

        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list)
        if args.prompt:
            assert self.mapping is not None
            self.label_to_word = eval(self.mapping)

            for key in self.label_to_word:
                if self.label_to_word[key][0] not in ['<', '[', '.', ',']:
                    print(tokenizer.tokenize(self.label_to_word[key]))
                    assert len(tokenizer.tokenize(' ' + self.label_to_word[key])) == 1 or len(tokenizer.tokenize(' ' + self.label_to_word[key])) == 2
                    self.label_to_word[key] = tokenizer._convert_token_to_id(tokenizer.tokenize(' ' + self.label_to_word[key])[-1])
                else:
                    self.label_to_word[key] = tokenizer.convert_tokens_to_ids(self.label_to_word[key])
                logger.info("Label {} to word {} ({})".format(key, tokenizer.convert_ids_to_tokens(self.label_to_word[key]), self.label_to_word[key]))
            
            if len(self.label_list) > 1:
                self.label_word_list = [self.label_to_word[label] for label in self.label_list]
            else:
                self.label_word_list = [self.label_to_word[label] for label in ['0', '1']]
        else:
            self.label_to_word = None
            self.label_word_list = None

        if (mode == "train") or not self.use_demo:
            self.num_sample = 1
        else:
            self.num_sample = args.num_sample

        if args.prompt and args.template_list is not None:
            logger.info("There are %d templates. Multiply num_sample by %d" % (len(args.template_list), len(args.template_list)))
            self.num_sample *= len(args.template_list)
                
        logger.info("Total num_sample for mode %s: %d" % (mode, self.num_sample))

        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else self.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode,
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
                args.task_name,
            ),
        )

        logger.info(f"Creating/loading examples from dataset file at {self.data_dir}")

        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.query_examples, self.sentence_pair_src, self.sentence_pair_dst = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {self.data_dir}")

                if mode == "dev":
                    self.query_examples = self.processor.get_dev_examples(self.data_dir)
                elif mode == "test":
                    self.query_examples = self.processor.get_test_examples(self.data_dir)
                elif mode == 'train':
                    self.query_examples = self.processor.get_train_examples(self.data_dir)
                else:
                    raise NotImplementedError()
                if self.args.enable_mlm_training:
                    if self.args.enable_src_lang_translate_test:
                        raise NotImplementedError()
                        
                    self.sentence_pair_src = self.processor.get_test_examples(args.src_data_dir)
                    self.sentence_pair_dst = self.processor.get_test_examples(args.dst_data_dir)
                else:
                    self.sentence_pair_src = None
                    self.sentence_pair_dst = None

                start = time.time()
                torch.save([self.query_examples, self.sentence_pair_src, self.sentence_pair_dst], cached_features_file)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

        if self.use_demo and args.demo_filter:
            split_name = ''
            if mode == 'train':
                split_name = 'train'
            elif mode == 'dev':
                if args.task_name == 'mnli':
                    split_name = 'dev_matched'
                elif args.task_name == 'mnli-mm':
                    split_name = 'dev_mismatched'
                else:
                    split_name = 'dev'
            elif mode == 'test':
                if args.task_name == 'mnli':
                    split_name = 'test_matched'
                elif args.task_name == 'mnli-mm':
                    split_name = 'test_mismatched'
                else:
                    split_name = 'test'
            else:
                raise NotImplementedError

            self.query_emb = np.load(os.path.join(self.data_dir, "{}_{}.npy".format(split_name, args.demo_filter_model)))
            logger.info("Load embeddings (for demonstration filtering) from {}".format(os.path.join(self.data_dir, "{}_{}.npy".format(split_name, args.demo_filter_model))))

            assert len(self.query_emb) == len(self.query_examples)
 
        self.features = []
        _ = 0
        for query_idx in tqdm(range(len(self.query_examples))):
            example = self.query_examples[query_idx]
            template = self.template

            self.features.append(self.convert_fn(
                example=example,
                use_demo=self.use_demo,
                label_list=self.label_list,
                prompt=args.prompt,
                template=template,
                label_word_list=self.label_word_list,
                verbose=True if _ == 0 else False,
            ))

            _ += 1
        
        if args.enable_mlm_training and mode =='train':
            assert len(self.sentence_pair_src) == len(self.sentence_pair_dst)
            num_mlm_sample = int(len(self.features) * args.mlm_sample_rate)
            indices_src = np.arange(len(self.sentence_pair_src))
            np.random.shuffle(indices_src)
            picked_src_sample = indices_src[:num_mlm_sample]

            indices_dst = np.arange(len(self.sentence_pair_dst))
            np.random.shuffle(indices_dst)
            picked_dst_sample = indices_dst[:num_mlm_sample]

            for i in range(len(picked_src_sample)):
                src_text = self.sentence_pair_src[picked_src_sample[i]]
                dst_text = self.sentence_pair_dst[picked_dst_sample[i]]

                inputs = tokenize_multipart_input(
                    input_text_list=(src_text.text_a, dst_text.text_a),
                    max_length=self.args.max_seq_length,
                    tokenizer=self.tokenizer,
                    task_name=self.args.task_name,
                    prompt=False,
                    need_padding=False
                )
                input_ids = torch.tensor(inputs['input_ids'])
                attention_mask = inputs['attention_mask']

                input_ids, labels = self.torch_mask_tokens(input_ids)
                input_ids = input_ids.tolist()
                labels = labels.tolist()
                input_ids, attention_mask, labels = paddings(input_ids, max_length=self.args.max_seq_length, tokenizer=self.tokenizer, attention_mask=attention_mask, labels=labels)

                feat = OurInputFeatures(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    label=labels,
                    is_mlm_case=True,
                    mask_pos=[-1],
                    label_word_list=[-1, -1, -1, -1, -1]
                )
                self.features.append(feat)

            random.shuffle(self.features)
        self.size = len(self.features)

    def select_context(self, context_examples):
        max_demo_per_label = 1
        counts = {k: 0 for k in self.label_list}
        if len(self.label_list) == 1:
            counts = {'0': 0, '1': 0}
        selection = []

        if self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail:
            order = np.random.permutation(len(context_examples))
            for i in range(min(self.args.gpt3_in_context_num, len(order))):
                selection.append(context_examples[order[i]])
        else:
            order = np.random.permutation(len(context_examples))

            for i in order:
                label = context_examples[i].label
                if len(self.label_list) == 1:
                    label = '0' if float(label) <= median_mapping[self.args.task_name] else '1'
                if counts[label] < max_demo_per_label:
                    selection.append(context_examples[i])
                    counts[label] += 1
                if sum(counts.values()) == len(counts) * max_demo_per_label:
                    break
        
            assert len(selection) > 0
        
        return selection

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if self.features is None:
            query_idx, context_indices, bootstrap_idx = self.example_idx[i]
            example = self.query_examples[query_idx]
            template = self.template

            features = self.convert_fn(
                example=example,
                use_demo=self.use_demo,
                label_list=self.label_list,
                prompt=self.args.prompt,
                template=template,
                label_word_list=self.label_word_list,
                verbose=False,
            )
        else:
            features = self.features[i]
            
        return features

    def get_labels(self):
        return self.label_list

    def torch_mask_tokens(self, inputs, special_tokens_mask=None):
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.args.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(labels, already_has_special_tokens=True)
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

    def convert_fn(
        self,
        example,
        use_demo=False,
        label_list=None,
        prompt=False,
        template=None,
        label_word_list=None,
        verbose=False
    ):

        max_length = self.args.max_seq_length    

        label_map = {label: i for i, label in enumerate(label_list)} 
        if len(label_list) == 1:
            label_map = {'0': 0, '1': 1}

        if example.label is None:
            example_label = None
        elif len(label_list) == 1:
            example_label = float(example.label)
        else:
            example_label = label_map[example.label]

        if self.only_fetch_label is not None:
            for_soft_label_flag = True
        else:
            for_soft_label_flag = False

        if not use_demo:
            if self.args.enable_two_tower_model:
                prompt_inputs = tokenize_multipart_input(
                    input_text_list=[],
                    max_length=self.prompt_length,
                    tokenizer=self.tokenizer,
                    task_name=self.args.task_name,
                    prompt=prompt,
                    template=template,
                    label_word_list=label_word_list,
                    first_sent_limit=self.args.first_sent_limit,
                    other_sent_limit=self.args.other_sent_limit,
                    label=example_label
                )
                prompt_input_ids = prompt_inputs['input_ids']
                prompt_attention_mask = prompt_inputs['attention_mask']
                mask_pos = prompt_inputs['mask_pos']
                label_word_list = prompt_inputs['label_word_list']
                label = prompt_inputs['label']

                original_inputs = tokenize_multipart_input(
                    input_text_list=input_example_to_tuple(example),
                    max_length=self.context_length,
                    tokenizer=self.tokenizer,
                    task_name=self.args.task_name,
                    prompt=False,
                )
                
                context_input_ids = original_inputs['input_ids']
                context_attention_mask = original_inputs['attention_mask']

                pad_label = label

                while len(pad_label) < self.args.max_seq_length:
                    pad_label.append(-100)

                features = OurInputFeatures(
                    input_ids=context_input_ids,
                    prompt_input_ids=prompt_input_ids,
                    attention_mask=context_attention_mask,
                    prompt_attention_mask=prompt_attention_mask,
                    label=pad_label,
                    mask_pos=mask_pos,
                    label_word_list=label_word_list,
                    is_mlm_case=False,
                    is_for_soft_label=for_soft_label_flag
                )

            else:
                inputs = tokenize_multipart_input(
                    input_text_list=input_example_to_tuple(example),
                    max_length=max_length,
                    tokenizer=self.tokenizer,
                    task_name=self.args.task_name,
                    prompt=prompt,
                    template=template,
                    label_word_list=label_word_list,
                    first_sent_limit=self.args.first_sent_limit,
                    other_sent_limit=self.args.other_sent_limit,
                    label=example_label
                )
                if self.args.prompt:
                    features = OurInputFeatures(**inputs, is_mlm_case=False, is_for_soft_label=for_soft_label_flag)
                else:
                    features = OurInputFeatures(**inputs)


        else:
            raise NotImplementedError()
            if self.args.double_demo:
                max_length = max_length * 2
            if self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail:
                max_length = 512
            augmented_example = []
            query_text = input_example_to_tuple(example) 
            support_by_label = [[] for i in range(len(label_map))]

            if self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail:
                support_labels = []
                augmented_example = query_text
                for support_example in supports:
                    augmented_example += input_example_to_tuple(support_example)
                    current_label = support_example.label
                    if len(label_list) == 1:
                        current_label = '0' if float(current_label) <= median_mapping[self.args.task_name] else '1' 
                    support_labels.append(label_map[current_label])
            else:
                for label_name, label_id in label_map.items():
                    if len(label_list) == 1:
                        for support_example in filter(lambda s: ('0' if float(s.label) <= median_mapping[self.args.task_name] else '1') == label_name, supports):
                            support_by_label[label_id] += input_example_to_tuple(support_example)
                    else:
                        for support_example in filter(lambda s: s.label == label_name, supports):
                            support_by_label[label_id] += input_example_to_tuple(support_example)

                augmented_example = query_text
                for label_id in range(len(label_map)):
                    augmented_example += support_by_label[label_id]

            inputs = tokenize_multipart_input(
                input_text_list=augmented_example,
                max_length=max_length,
                tokenizer=self.tokenizer,
                task_name=self.args.task_name,
                prompt=prompt,
                template=template,
                label_word_list=label_word_list,
                first_sent_limit=self.args.first_sent_limit,
                other_sent_limit=self.args.other_sent_limit,
                truncate_head=self.args.truncate_head,
                gpt3=self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail,
                support_labels=None if not (self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail) else support_labels
            )
            features = OurInputFeatures(**inputs, label=example_label, mask_lm_label=label_word_list[example_label])

        if verbose:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("features: %s" % features)
            logger.info("text: %s" % self.tokenizer.decode(features.input_ids))

        return features



