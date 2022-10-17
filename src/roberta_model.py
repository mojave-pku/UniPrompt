from transformers.modeling_roberta import RobertaEmbeddings, RobertaPooler, BaseModelOutputWithPooling, RobertaLayer, BaseModelOutput, RobertaLMHead, RobertaClassificationHead, RobertaPreTrainedModel
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.configuration_roberta import RobertaConfig
import torch.nn.functional as F


class RobertaModEmbeddings(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = RobertaEmbeddings(config)

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        token_type_ids=None,

    ):
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids
        )

        return embedding_output

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)


class RobertaLowerTower(nn.Module):
    def __init__(self, config, num_layers):
        super().__init__()
        self.config = config
        self.encoder = nn.ModuleList([RobertaLayer(config) for _ in range(num_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.encoder):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states


class RobertaHigherTower(nn.Module):
    def __init__(self, config, num_layers, add_pooling_layer=True):
        super().__init__()
        self.config = config
        self.encoder = nn.ModuleList([RobertaLayer(config) for _ in range(num_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.encoder):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        assert len(tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)) == 1
        return hidden_states


class RobertaTwoTowerModel(RobertaPreTrainedModel):
    def __init__(self, config, num_shared_layers):
        super().__init__(config)
        self.config = config
        
        self.roberta = RobertaModEmbeddings(config)
        self.context_tower = RobertaLowerTower(config, num_shared_layers)
        self.prompt_tower = RobertaLowerTower(config, num_shared_layers)
        self.high_tower = RobertaHigherTower(config, 12-num_shared_layers)

        self.pooler = RobertaPooler(config)


        self.lm_head = RobertaLMHead(config)

        self.init_weights()
    
    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        prompt_input_ids=None,
        prompt_attention_mask=None,
        mask_pos=None,
        labels=None,
        label_word_list=None,
        mask_lm_label=None,
        is_mlm_case=None,
        is_for_soft_label=None

    ):
        output_attentions = self.config.output_attentions
        output_hidden_states = (
            self.config.output_hidden_states
        )
        return_dict = self.config.use_return_dict

        input_shape = input_ids.size()
        prompt_input_shape = prompt_input_ids.size()

        device = input_ids.device 

        assert attention_mask is not None
        assert prompt_attention_mask is not None

        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        prompt_extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(prompt_attention_mask, prompt_input_shape, device)

        encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(None, self.config.num_hidden_layers)
        

        batch_size = input_ids.size(0)
        sequence_length = input_ids.size(-1) + prompt_input_ids.size(-1)
        label_word_list = label_word_list[0]  
        labels = labels.long()

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        embedding_output = self.roberta(
            input_ids=input_ids, position_ids=None, 
        )
        prompt_embedding_output = self.roberta(
            input_ids=prompt_input_ids, position_ids=None,
        )

        context_lower_encoder_outputs = self.context_tower(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
        )
        prompt_lower_encoder_outputs = self.prompt_tower(
            prompt_embedding_output,
            attention_mask=prompt_extended_attention_mask,
            head_mask=head_mask,
        )

        joint_encoder_outputs = torch.cat(
            [prompt_lower_encoder_outputs, context_lower_encoder_outputs],
            dim=1
        )
        joint_attention_mask = torch.cat(
            [prompt_extended_attention_mask, extended_attention_mask],
            dim=-1
        )

        sequence_output = self.high_tower(
            joint_encoder_outputs,
            attention_mask=joint_attention_mask,
            head_mask=head_mask,
        )

        if is_for_soft_label[0]: 
            sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]
            return sequence_mask_output

        prediction_mask_scores = self.lm_head(sequence_output)
        vocab_size = prediction_mask_scores.size()[-1]

        final_prediction_logits = torch.zeros(batch_size, sequence_length, vocab_size).cuda()
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