import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.modeling_bert import BertPreTrainedModel, BertForSequenceClassification, BertModel, BertOnlyMLMHead
from transformers.modeling_roberta import RobertaPreTrainedModel, RobertaForSequenceClassification, RobertaModel, RobertaLMHead, RobertaClassificationHead
from transformers.modeling_outputs import SequenceClassifierOutput

import logging
logger = logging.getLogger(__name__)

def resize_token_type_embeddings(model, new_num_types: int, random_segment: bool):
    if hasattr(model, 'bert'):
        old_token_type_embeddings = model.bert.embeddings.token_type_embeddings
    else:
        raise NotImplementedError
    new_token_type_embeddings = nn.Embedding(new_num_types, old_token_type_embeddings.weight.size(1))
    if not random_segment:
        new_token_type_embeddings.weight.data[:old_token_type_embeddings.weight.size(0)] = old_token_type_embeddings.weight.data

    model.config.type_vocab_size = new_num_types
    if hasattr(model, 'bert'):
        model.bert.embeddings.token_type_embeddings = new_token_type_embeddings
    else:
        raise NotImplementedError


class BertForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

        self.model_args = None
        self.data_args = None

        self.lb = None
        self.ub = None

        self.return_full_softmax = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        labels=None,
        label_word_list=None
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        prediction_mask_scores = self.cls(sequence_mask_output)

        if self.return_full_softmax:
            if labels is not None:
                return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
            return prediction_mask_scores

        logits = []
        for label_id in range(len(label_word_list)):
            logits.append(prediction_mask_scores[:, label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits) 

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        return ((loss,) + output) if loss is not None else output


class RobertaForPromptFinetuning(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.lm_head = RobertaLMHead(config)
        self.init_weights()

        self.model_args = None
        self.data_args = None

        self.lb = None
        self.ub = None

        self.return_full_softmax = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
        label_word_list=None,
        mask_lm_label=None,
        is_mlm_case=None,
        is_for_soft_label=None
    ):
        batch_size = input_ids.size(0)
        sequence_length = input_ids.size(-1)
        label_word_list = label_word_list[0]  
        labels = labels.long()

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask
        )

        sequence_output, pooled_output = outputs[:2]
        if is_for_soft_label[0]: 
            sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]
            return sequence_mask_output

        prediction_mask_scores = self.lm_head(sequence_output)
        vocab_size = prediction_mask_scores.size()[-1]

        final_prediction_logits = torch.zeros(batch_size, sequence_length,vocab_size).cuda()
        for i in range(batch_size):
            if not is_mlm_case[i]:
                current_pred_mask_scores = prediction_mask_scores[i]
                logits = []
                for label_id in range(len(label_word_list)):
                    if batch_size == 1:
                        m_pos = mask_pos
                    else:
                        m_pos = mask_pos[i]
                    logits.append(current_pred_mask_scores[m_pos, label_word_list[label_id]].unsqueeze(-1))
                logits = torch.cat(logits, -1)
                logits_after_log_softmax = F.log_softmax(logits, -1)

                mlm_logits = torch.zeros(vocab_size).cuda()

                for label_id in range(len(label_word_list)):
                    mlm_logits[label_word_list[label_id]] = logits_after_log_softmax[label_id]
                final_prediction_logits[i, m_pos] = mlm_logits
            else:
                final_prediction_logits[i] = F.log_softmax(prediction_mask_scores[i], -1)

        loss_fct = nn.NLLLoss(ignore_index=-100)
        loss = loss_fct(final_prediction_logits.view(-1, final_prediction_logits.size(-1)), labels.view(-1))

        label_mask = torch.ones(1, vocab_size, dtype=torch.bool).cuda()
        for label_id in label_word_list:
            label_mask[0][label_id] = False

        prediction_logits_for_matrics = torch.clone(final_prediction_logits)
        for i in range(batch_size):
            if not is_mlm_case[i]:
                prediction_logits_for_matrics[i] = final_prediction_logits[i].masked_fill(label_mask, -float("inf"))

        prediction = torch.argmax(prediction_logits_for_matrics, -1)
    
        output = (prediction,)
      
        return ((loss,) + output) if loss is not None else output

    def get_output_embeddings(self):
        return self.lm_head.decoder
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings
