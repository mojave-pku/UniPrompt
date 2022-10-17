import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
import torch

import numpy as np

import transformers
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import HfArgumentParser, TrainingArguments, set_seed
from transformers import data

from src.dataset import FewShotDataset
from src.models import BertForPromptFinetuning, RobertaForPromptFinetuning, resize_token_type_embeddings
from src.trainer import Trainer
from src.processors import processors_mapping, num_labels_mapping, output_modes_mapping, compute_metrics_mapping, bound_mapping

from filelock import FileLock
from datetime import datetime

from copy import deepcopy
from tqdm import tqdm
import json

from src.roberta_model import RobertaTwoTowerModel
from src.state_dict_transfer import update_state_dict_for_two_tower

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    few_shot_type: str = field(
        default='prompt-demo',
        metadata={"help": "Few-shot learning model type. Choice: finetune, prompt, prompt-demo"}
    )

    random_segment: bool = field(
        default=False,
        metadata={"help": "Whether to reinitialize the token type embeddings (only for BERT)."}
    )

@dataclass
class DynamicDataTrainingArguments(DataTrainingArguments):
    src_lang: str = field(
        default=None,
        metadata={"help": "Language (for record results)"}
    )

    dst_lang: str = field(
        default=None,
        metadata={"help": "Language (for record results)"}
    )

    src_data_dir: str = field(
        default=None,
        metadata={"help": "Language (for record results)"}
    )

    dst_data_dir: str = field(
        default=None,
        metadata={"help": "Language (for record results)"}
    )

    num_k: Optional[int] = field(
        default=16,
        metadata={"help": "Number of training instances per class"}
    )

    num_sample: Optional[int] = field(
        default=16,
        metadata={"help": "Number of samples (for inference) in fine-tuning with demonstrations"}
    )

    num_demo: Optional[int] = field(
        default=1,
        metadata={"help": "Number of demonstrations from each class"}
    )

    auto_demo: bool = field(
        default=True,
        metadata={"help": "Automatically generate template for using demonstrations"}
    )

    src_template: str = field(
        default=None,
        metadata={"help": "Template"}
    )

    dst_template: str = field(
        default=None,
        metadata={"help": "Template"}
    )

    src_mapping: str = field(
        default=None,
        metadata={"help": "Label word mapping"}
    )

    dst_mapping: str = field(
        default=None,
        metadata={"help": "Label word mapping"}
    )

    template_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the templates, one per line. Do not set this when prompt_path is used"}
    )

    mapping_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the label word mappings, one per line. Do not set this when prompt_path is used"}
    )

    prompt_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the prompts (templates and mappings), one per line"}
    )
 
    template_id: int = field(
        default=None,
        metadata={"help": "Template id if using template_path"}
    )

    mapping_id: int = field(
        default=None,
        metadata={"help": "Mapping id if using template_path"}
    )

    prompt_id: int = field(
        default=None,
        metadata={"help": "Prompt id if using prompt_path"}
    )

    top_n_template: int = field(
        default=None,
        metadata={"help": "Use top-n template in the template path"}
    )

    enable_two_tower_model: bool = field(
        default=False,
        metadata={"help": "Whether to reinitialize the token type embeddings (only for BERT)."}
    )

    prompt_tower_init_type: str = field(
        default='xlmr',
        metadata={"help": "Whether to reinitialize the token type embeddings (only for BERT)."}
    )

    prompt_tower_init_state_dict: str = field(
        default=None,
        metadata={"help": "单独为prompt tower使用的state dict. "}
    )

    state_dict_path: str = field(
        default='',
        metadata={"help": "Whether to reinitialize the token type embeddings (only for BERT)."}
    )

    num_shared_layers: int = field(
        default=8,
        metadata={"help": "Whether to reinitialize the token type embeddings (only for BERT)."}
    )

    enable_soft_label_words: bool = field(
        default=False,
        metadata={"help": "是否使用soft label words."}
    )

    soft_label_init: bool = field(
        default=False,
        metadata={"help": "是否使用hard label words 初始化 soft label words."}
    )

    dynamic_soft_label_init: bool = field(
        default=False,
        metadata={"help": "是否使用动态初始化 soft label words."}
    )

    enable_soft_label_evaluate: bool = field(
        default=False,
        metadata={"help": "是否评估soft label的质量"}
    )

    soft_label_noise_seed: int = field(
        default=-1,
        metadata={"help": "用来扰动soft label的noise"}
    )



    tag: str = field(
        default='',
        metadata={"help": "Set the tag and find the result easier in the log."}
    )

    demo_filter: bool = field(
        default=False,
        metadata={"help": "Only use similar instances in demonstrations"}
    )

    demo_filter_rate: float = field(
        default=0.5,
        metadata={"help": "Only use top-x\% similar instances in demonstrations"}
    )

    demo_filter_model: str = field(
        default=None,
        metadata={"help": "Model name for demonstration filter embeddings. Will load embeddings based on the model name."}
    )

    debug_mode: bool = field(
        default=False,
        metadata={"help": "Debug mode"}
    )

    double_demo: bool = field(
        default=False,
        metadata={"help": "Use double length for using demonstrations"}
    )

    first_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of the first sentence (i.e., sent_0)"}
    )

    other_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of sentences other than the first sentence"}
    )

    use_full_length: bool = field(
        default=None,
        metadata={"help": "Use the full length (512)"}
    )

    gpt3_in_context_head: bool = field(
        default=False,
        metadata={"help": "GPT-3's in-context learning (context at the beginning)"}
    )

    gpt3_in_context_tail: bool = field(
        default=False,
        metadata={"help": "GPT-3's in-context learning (context at the end)"}
    )

    gpt3_in_context_num: int = field(
        default=32,
        metadata={"help": "Number of context examples"}
    )

    truncate_head: bool = field(
        default=False,
        metadata={"help": "When exceeding the maximum length, truncate the head instead of the tail."}
    )

    prompt: bool = field(
        default=False,
        metadata={"help": "Whether to use prompt-based fine-tuning"}
    )
    template_list: list = field(
        default=None,
        metadata={"help": "(DO NOT List of templates (only initialized after the program starts."}
    )

    result_file_path: str = field(
        default='',
        metadata={"help": "Set the tag and find the result easier in the log."}
    )

    note: str = field(
        default='',
        metadata={"help": "Set the tag and find the result easier in the log."}
    )

    enable_mlm_training: bool = field(
        default=False,
        metadata={"help": "Set the tag and find the result easier in the log."}
    )

    enable_dst_lang_text: bool = field(
        default=False,
        metadata={"help": "Set the tag and find the result easier in the log."}
    )

    enable_src_lang_translate_test: bool = field(
        default=False,
        metadata={"help": "Language (for record results)"}
    )

    mlm_sample_rate: float = field(
        default=1.0,
        metadata={"help": "Set the tag and find the result easier in the log."}
    )

    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Set the tag and find the result easier in the log."}
    )

    enable_soft_prompt: bool = field(
        default=False,
        metadata={"help": "Set the tag and find the result easier in the log."}
    )

    soft_prompt_init: bool = field(
        default=False,
        metadata={"help": "使用hard prompt的embedding初始化soft prompt."}
    )

    num_prompt_tokens: int = field(
        default=2,
        metadata={"help": "soft prompt token个数"}
    )



@dataclass
class DynamicTrainingArguments(TrainingArguments):
    array_id: int = field(
        default=-1,
        metadata={"help": "Array ID (contains seed and hyper-paramter search) to idenfity the model"}
    )

    model_id: int = field(
        default=-1,
        metadata={"help": "Model ID (contains template information) to identify the model"}
    )

    save_logit: bool = field(
        default=False,
        metadata={"help": "Save test file logit with name $TASK-$MODEL_ID-$ARRAY_ID.npy"}
    )

    save_logit_dir: str = field(
        default=None,
        metadata={"help": "Where to save the prediction result"}
    )

    fix_layers: int = field(
        default=0,
        metadata={"help": "Fix bottom-n layers when optimizing"}
    )

    fix_except_prompt: bool = field(
        default=False,
        metadata={"help": "Fix bottom-n layers when optimizing"}
    )


    save_at_last: bool = field(
        default=False,
        metadata={"help": "Instead of saving the best (dev performance) checkpoint, save the last checkpoint"}
    )

    no_train: bool = field(
        default=False,
        metadata={"help": "No training"}
    )
    no_predict: bool = field(
        default=False,
        metadata={"help": "No test"}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments))


    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if 'prompt' in model_args.few_shot_type:
        data_args.prompt = True

    if training_args.no_train:
        training_args.do_train = False
    if training_args.no_predict:
        training_args.do_predict = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    def load_template_and_mapping(prompt_path=None, prompt_id=None, template_path=None, template_id=None, mapping_path=None, mapping_id=None):
        if data_args.prompt:
            if prompt_path is not None:
                assert prompt_id is not None
                prompt_list = []
                with open(prompt_path) as f:
                    for line in f:
                        line = line.strip()
                        template, mapping = line.split('\t')
                        prompt_list.append((template, mapping))

                logger.info("Specify load the %d-th prompt: %s | %s" % (prompt_id, prompt_list[data_args.prompt_id][0], prompt_list[data_args.prompt_id][1]))
                return prompt_list[data_args.prompt_id] 
            else:
                if template_path is not None:
                    with open(template_path) as f:
                        template_list = []
                        for line in f:
                            line = line.strip()
                            if len(line) > 0:
                                template_list.append(line)

                    if data_args.top_n_template is not None:
                        template_list = template_list[:data_args.top_n_template]
                    logger.info("Load top-%d templates from %s" % (len(template_list), template_path))

                    if template_id is not None:
                        template = template_list[template_id]
                        template_list = None
                        logger.info("Specify load the %d-th template: %s" % (template_id, template))

                if mapping_path is not None:
                    assert mapping_id is not None 
                    with open(mapping_path) as f:
                        mapping_list = []
                        for line in f:
                            line = line.strip()
                            mapping_list.append(line)

                    mapping = mapping_list[mapping_id]
                    logger.info("Specify using the %d-th mapping: %s" % (mapping_id, mapping))


    try:
        num_labels = num_labels_mapping[data_args.task_name]
        output_mode = output_modes_mapping[data_args.task_name]
        logger.info("Task name: {}, number of labels: {}, output mode: {}".format(data_args.task_name, num_labels, output_mode))
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    def generate_template(template=None, template_list=None):
        if data_args.auto_demo and model_args.few_shot_type == 'prompt-demo':
            if data_args.gpt3_in_context_head or data_args.gpt3_in_context_tail: 
                logger.info("Automatically convert the template to GPT-3's in-context learning.")
                assert template_list is None

                old_template = template
                new_template = old_template + ''
                old_template = old_template.replace('*cls*', '')
                sent_num = 1
                if "_1" in old_template:
                    sent_num = 2
                for instance_id in range(data_args.gpt3_in_context_num):
                    sub_template = old_template + ''
                    for sent_id in range(sent_num):
                        sub_template = sub_template.replace("_{}*".format(sent_id), "_{}*".format(sent_num + sent_num * instance_id + sent_id))
                    sub_template = sub_template.replace("*mask*", "*labelx_{}*".format(instance_id))
                    if data_args.gpt3_in_context_tail:
                        new_template = new_template + sub_template 
                    else:
                        new_template = sub_template + new_template 
                logger.info("| {} => {}".format(template, new_template))
                return new_template
            else:
                logger.info("Automatically convert the template to using demonstrations.")
                if template_list is not None:
                    for i in range(len(template_list)):
                        old_template = template_list[i]
                        new_template = old_template + ''
                        old_template = old_template.replace('*cls*', '')
                        sent_num = 1
                        if "_1" in old_template:
                            sent_num = 2
                        for label_id in range(num_labels):
                            sub_template = old_template + ''
                            for sent_id in range(sent_num):
                                sub_template = sub_template.replace("_{}*".format(sent_id), "_{}*".format(sent_num + sent_num * label_id + sent_id))
                            sub_template = sub_template.replace("*mask*", "*label_{}*".format(label_id))
                            new_template = new_template + sub_template
                        logger.info("| {} => {}".format(template_list[i], new_template))
                        template_list[i] = new_template
                else:
                    old_template = template
                    new_template = old_template + ''
                    old_template = old_template.replace('*cls*', '')
                    sent_num = 1
                    if "_1" in old_template:
                        sent_num = 2
                    for label_id in range(num_labels):
                        sub_template = old_template + ''
                        for sent_id in range(sent_num):
                            sub_template = sub_template.replace("_{}".format(sent_id), "_{}".format(sent_num + sent_num * label_id + sent_id))
                        sub_template = sub_template.replace("*mask*", "*label_{}*".format(label_id))
                        new_template = new_template + sub_template
                    logger.info("| {} => {}".format(template, new_template))
                    return new_template
        else:
            return template
    data_args.src_template = generate_template(data_args.src_template)
    data_args.dst_template = generate_template(data_args.dst_template)

    if data_args.enable_soft_prompt:
        prompt_tokens = [f"[p{i}]" for i in range(data_args.num_prompt_tokens)]
        if data_args.enable_two_tower_model:
            new_template = f"*cls*{' '.join(prompt_tokens)}*mask*._*sep+*"
        else:
            new_template = f"*cls*{' '.join(prompt_tokens)}*mask*._*sent_0**sep+*"
        data_args.src_template = new_template
        data_args.dst_template = new_template
    
    if data_args.enable_soft_label_words:
        new_mapping = '{1: "[t0]", 2: "[t1]", 3: "[t2]", 4: "[t3]", 5: "[t4]"}'
        data_args.src_mapping = new_mapping
        data_args.dst_mapping = new_mapping
    
    print(data_args)

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists.")

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)
    torch.backends.cudnn.deterministic = True
    

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    if 'prompt' in model_args.few_shot_type:
        if config.model_type == 'roberta' or config.model_type == 'xlm-roberta':
            if data_args.enable_two_tower_model:
                model_fn = RobertaTwoTowerModel
            else:
                model_fn = RobertaForPromptFinetuning
        elif config.model_type == 'bert':
            model_fn = BertForPromptFinetuning
        else:
            raise NotImplementedError
    elif model_args.few_shot_type == 'finetune':
        model_fn = AutoModelForSequenceClassification
    else:
        raise NotImplementedError
    special_tokens = []

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        additional_special_tokens=special_tokens,
        cache_dir=model_args.cache_dir,
    )

    if data_args.enable_soft_prompt:
        num_added_tokens = tokenizer.add_tokens([f"[p{i}]" for i in range(data_args.num_prompt_tokens)])
        logger.info(f"add {num_added_tokens} tokens for soft prompt.")
        assert len(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' '.join([f"[p{i}]" for i in range(data_args.num_prompt_tokens)])))) == data_args.num_prompt_tokens

    if data_args.enable_soft_label_words:
        num_added_tokens = tokenizer.add_tokens([f"[t{i}]" for i in range(5)])
        logger.info(f"add {num_added_tokens} tokens for soft label words.")
        assert len(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' '.join([f"[t{i}]" for i in range(5)])))) == 5

    train_dataset = (
        FewShotDataset(data_args, tokenizer=tokenizer, mode="train", use_demo=("demo" in model_args.few_shot_type))
    )
    eval_dataset = (
        FewShotDataset(data_args, tokenizer=tokenizer, mode="dev", use_demo=("demo" in model_args.few_shot_type))
        if training_args.do_eval
        else None
    )

    test_dataset = (
        FewShotDataset(data_args, tokenizer=tokenizer, mode="test", use_demo=("demo" in model_args.few_shot_type))
        if training_args.do_predict
        else None
    )

    set_seed(training_args.seed)
    if data_args.enable_two_tower_model:
        state_dict = update_state_dict_for_two_tower(
        data_args.state_dict_path, 
        prompt_tower_init_type=data_args.prompt_tower_init_type, 
        prompt_tower_init_state_dict=data_args.prompt_tower_init_state_dict,
        low_tower_layers=data_args.num_shared_layers
        )

        model = model_fn.from_pretrained(
            pretrained_model_name_or_path=None,
            state_dict=state_dict,
            from_tf=False,
            config=config,
            cache_dir=model_args.cache_dir,
            num_shared_layers=data_args.num_shared_layers
        )    
    else:
        model = model_fn.from_pretrained(
            model_args.model_name_or_path,
            from_tf=False,
            config=config,
            cache_dir=model_args.cache_dir,
        )    

    if data_args.enable_soft_prompt or data_args.enable_soft_label_words:
        model.resize_token_embeddings(len(tokenizer))

    if data_args.enable_soft_prompt:
        if data_args.soft_prompt_init:
            assert data_args.num_prompt_tokens == 2
            old_embeddings = model.roberta.embeddings.word_embeddings
            assert data_args.src_lang == 'en'  
            target_token_ids = tokenizer.convert_tokens_to_ids(['It', 'is'])
            prompt_token_ids = tokenizer.convert_tokens_to_ids(['[p0]', '[p1]'])
            assert len(target_token_ids) == len(prompt_token_ids)
            for i in range(len(target_token_ids)):
                target_idx = target_token_ids[i]
                prompt_idx = prompt_token_ids[i]
                old_embeddings.weight.data[prompt_idx, :] = old_embeddings.weight.data[target_idx, :]

    if data_args.enable_soft_label_words:
        if data_args.soft_label_init:
            mappings = {1: "bad", 2: "fine", 3: "funny", 4: "perfect", 5: "wonderful"}
            label_tokens = [mappings[i+1] for i in range(5)]
            target_label_token_ids = tokenizer.convert_tokens_to_ids(label_tokens)
            prompt_label_token_ids = tokenizer.convert_tokens_to_ids([f"[t{i}]" for i in range(5)])
            assert len(target_label_token_ids) == len(prompt_label_token_ids)
            old_lm_head_weight = model.lm_head.decoder.weight.data
            old_lm_head_weight.permute(1, 0)
            for i in range(len(target_label_token_ids)):
                target_idx = target_label_token_ids[i]
                label_word_idx = prompt_label_token_ids[i]
                old_lm_head_weight[label_word_idx, :] = old_lm_head_weight[target_idx, :]
            old_lm_head_weight.permute(0, 1)
            model.lm_head.decoder.weight.data = old_lm_head_weight


    if config.model_type == 'bert':
        model.resize_token_embeddings(len(tokenizer))
        resize_token_type_embeddings(model, new_num_types=10, random_segment=model_args.random_segment)

    if output_modes_mapping[data_args.task_name] == 'regression':
        model.lb, model.ub = bound_mapping[data_args.task_name]
    model.model_args = model_args
    model.data_args = data_args
    model.tokenizer = tokenizer

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            preds = p.predictions

            label_ids = p.label_ids

            return compute_metrics_mapping[task_name](task_name, preds, label_ids)

        return compute_metrics_fn
    

    if data_args.dynamic_soft_label_init:
        assert data_args.soft_label_init is False

        trainer_for_soft_label_init = Trainer(
            model=model,
            args=training_args,
        )

        def generate_dynamic_soft_label_init(label):
            specfic_label_dataset = FewShotDataset(data_args, tokenizer=tokenizer, mode="train", use_demo=("demo" in model_args.few_shot_type), only_fetch_label=label)
            soft_label_embeddings = trainer_for_soft_label_init.prepare_for_soft_labels(eval_dataset=specfic_label_dataset)
            return soft_label_embeddings

        all_soft_label_embeddings = []
        for i in range(5): 
            label_id = i + 1
            all_soft_label_embeddings.append(generate_dynamic_soft_label_init(label_id))
        
        
        old_lm_head_weight = model.lm_head.decoder.weight.data
        old_lm_head_weight.permute(1, 0)
        prompt_label_token_ids = tokenizer.convert_tokens_to_ids([f"[t{i}]" for i in range(5)])
        assert len(prompt_label_token_ids) == 5
        for i in range(5):
            label_word_idx = prompt_label_token_ids[i]
            old_lm_head_weight[label_word_idx, :] = all_soft_label_embeddings[i]
        old_lm_head_weight.permute(0, 1)
        model.lm_head.decoder.weight.data = old_lm_head_weight    

    if data_args.enable_soft_label_evaluate:

        trainer_for_gold_soft_label_init = Trainer(
            model=model,
            args=training_args,
        )

        def generate_gold_dynamic_soft_label_init(label):
            specfic_label_dataset = FewShotDataset(data_args, tokenizer=tokenizer, mode="train", use_demo=("demo" in model_args.few_shot_type), only_fetch_label=label)
            soft_label_embeddings = trainer_for_gold_soft_label_init.prepare_for_soft_labels(eval_dataset=specfic_label_dataset)
            return soft_label_embeddings

        gold_soft_embeddings = []
        for i in range(5): 
            label_id = i + 1
            gold_soft_embeddings.append(generate_gold_dynamic_soft_label_init(label_id))
        
        original_prompt_label_token_ids = tokenizer.convert_tokens_to_ids([f"[t{i}]" for i in range(5)])

        soft_label_words_scores = []
        for i in range(5):
            current_label_token_ids = original_prompt_label_token_ids[i]
            current_label_token_embeddings = model.lm_head.decoder.weight.data[current_label_token_ids]
            current_score = torch.square(gold_soft_embeddings[i] - current_label_token_embeddings).sum()
            soft_label_words_scores.append(current_score)
        final_label_words_scores = sum(soft_label_words_scores)
        
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name)
    )

    if training_args.do_train:
        trainer.train(model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None)
        if training_args.save_at_last:
            trainer.save_model(training_args.output_dir)
 
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)
            torch.save(model_args, os.path.join(training_args.output_dir, "model_args.bin"))
            torch.save(data_args, os.path.join(training_args.output_dir, "data_args.bin"))
        
        if data_args.enable_two_tower_model:
            model = model_fn.from_pretrained(training_args.output_dir, num_shared_layers=data_args.num_shared_layers)
        else:
            model = model_fn.from_pretrained(training_args.output_dir)
        model = model.to(training_args.device)
        trainer.model = model

        if output_modes_mapping[data_args.task_name] == 'regression':
            model.lb, model.ub = bound_mapping[data_args.task_name]
        model.model_args = model_args
        model.data_args = data_args
        model.tokenizer = tokenizer

    final_result = {
        'time': str(datetime.today()),
    }

    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Validate ***")

        eval_datasets = [eval_dataset]

        for eval_dataset in eval_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            output = trainer.evaluate(eval_dataset=eval_dataset)
            eval_result = output.metrics 

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
                        final_result[eval_dataset.args.task_name + '_dev_' + key] = value
            eval_results.update(eval_result)

    test_results = {}
    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            test_datasets.append(
                FewShotDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test", use_demo=('demo' in model_args.few_shot_type))
            )

        for test_dataset in test_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(test_dataset.args.task_name)
            output = trainer.evaluate(eval_dataset=test_dataset)
            test_result = output.metrics

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    for key, value in test_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
                        final_result[test_dataset.args.task_name + '_test_' + key] = value
                with open(data_args.result_file_path, 'a') as f:
                    result_str = [f"{k}: {test_result[k]}" for k in test_result.keys()]
                    if data_args.enable_soft_label_evaluate:
                        f.write(f"|{data_args.note}|{model_args.model_name_or_path}|{data_args.src_lang}->{data_args.dst_lang}|{training_args.seed}|{test_result['eval_acc']}|{final_label_words_scores}\n")
                    else:
                        f.write(f"|{data_args.note}|{model_args.model_name_or_path}|{data_args.src_lang}->{data_args.dst_lang}|{training_args.seed}|{test_result['eval_acc']}\n")

                if training_args.save_logit:
                    predictions = output.predictions
                    num_logits = predictions.shape[-1]
                    logits = predictions.reshape([test_dataset.num_sample, -1, num_logits]).mean(axis=0)
                    np.save(os.path.join(training_args.save_logit_dir, "{}-{}-{}.npy".format(test_dataset.task_name, training_args.model_id, training_args.array_id)), logits)

            test_results.update(test_result)

    with FileLock('log.lock'):
        with open('log', 'a') as f:
            final_result.update(vars(model_args))
            final_result.update(vars(training_args))
            final_result.update(vars(data_args))
            if 'evaluation_strategy' in final_result:
                final_result.pop('evaluation_strategy')
            f.write(str(final_result) + '\n')
    
    return eval_results

if __name__ == "__main__":
    main()
