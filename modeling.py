from typing import Optional, Tuple, Union

import torch
from arg import args
from cornet import CorNet
from model_loss import FocalLoss, compute_kl_loss, multilabel_categorical_crossentropy
from model_outputs import MultiTaskBertForSequenceAndTokenClassifierOutput
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

if args.loss_type is None and args.problem_type == "single_label_classification":
    loss_fct = CrossEntropyLoss(weight=args.label_weights)
elif args.loss_type is None and args.problem_type == "mulit_label_classification":
    loss_fct = BCEWithLogitsLoss()
elif (
    args.loss_type == "focal_loss"
    and args.problem_type == "single_label_classification"
):
    loss_fct = FocalLoss(alpha=0.25, gamma=2, activation_type="softmax")
elif (
    args.loss_type == "focal_loss" and args.problem_type == "mulit_label_classification"
):
    loss_fct = FocalLoss(
        alpha=0.25,
        gamma=2,
        activation_type="sigmoid",
    )


class CustomerBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.loss_type = config.args.loss_type
        self.use_r_drop = config.args.use_r_drop
        self.bert = BertModel(config)
        self.label_weights = config.args.label_weights
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                if self.loss_type == "FOCAL":
                    loss_fct = FocalLoss(alpha=0.25, gamma=2)
                elif self.loss_type == "CE":
                    loss_fct = CrossEntropyLoss()
                elif self.loss_type == "weightCE":
                    loss_fct = CrossEntropyLoss(weight=self.label_weights)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                if not self.use_r_drop:
                    if self.loss_type == "ZLPR":
                        loss = multilabel_categorical_crossentropy(logits, labels)
                    elif self.loss_type == "BCE":
                        loss_fct = BCEWithLogitsLoss()
                        loss = loss_fct(logits, labels)
                    else:
                        raise ValueError("loss_type must be ZLPR or BCE")
                else:
                    logits1, logits2 = logits[::2], logits[1::2]
                    if self.loss_type == "ZLPR":
                        # cross entropy loss for classifier
                        ce_loss = multilabel_categorical_crossentropy(
                            logits1, labels
                        ) + multilabel_categorical_crossentropy(logits2, labels)
                        kl_loss = compute_kl_loss(logits1, logits2)
                        # carefully choose hyper-parameters
                        loss = ce_loss + kl_loss * 2
                    elif self.loss_type == "BCE":
                        loss_fct = BCEWithLogitsLoss()
                        ce_loss = 4 * (
                            loss_fct(logits1, labels) + loss_fct(logits2, labels)
                        )
                        kl_loss = compute_kl_loss(logits1, logits2)
                        # carefully choose hyper-parameters
                        loss = ce_loss + kl_loss
                    else:
                        raise ValueError("loss_type must be ZLPR or BCE")
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class CorNetBertXML(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.cornet = CorNet(config.num_labels, corent_dim=1000, n_cornet_blocks=2)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        raw_logits = self.classifier(pooled_output)

        logits = self.cornet(raw_logits)
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MultiTaskBertForSequenceAndTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_token_labels = config.num_token_labels
        self.config = config
        self.loss_type = config.args.loss_type
        # self.use_r_drop = config.args.use_r_drop
        self.bert = BertModel(config)
        # self.label_weights = config.args.label_weights
        self.label_weights = None
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.token_classifier = nn.Linear(config.hidden_size, config.num_token_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        token_labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[
        Union[Tuple[torch.Tensor], SequenceClassifierOutput],
        Union[Tuple[torch.Tensor], TokenClassifierOutput],
    ]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        token_labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # print("outputs")
        # print(outputs)

        sequence_output, pooled_output = outputs[:2]

        # For sequence classification
        pooled_output = self.dropout(pooled_output)
        cls_logits = self.classifier(pooled_output)
        # print("pooled_output")
        # print(pooled_output)
        # print(pooled_output.shape)
        # print("cls_logits")
        # print(cls_logits)
        # print(cls_logits.shape)

        # For token classification
        sequence_output = self.dropout(sequence_output)
        token_logits = self.token_classifier(sequence_output)
        # print("sequence_output")
        # print(sequence_output)
        # print(sequence_output.shape)
        # print("token_logits")
        # print(token_logits)
        # print(token_logits.shape)

        total_loss, cls_loss, token_loss = None, None, None

        # Loss
        if labels is not None and token_labels is not None:
            # For sequence cls loss
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    cls_loss = loss_fct(cls_logits.squeeze(), labels.squeeze())
                else:
                    cls_loss = loss_fct(cls_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                if self.loss_type == "FOCAL":
                    loss_fct = FocalLoss(alpha=0.25, gamma=2)
                elif self.loss_type == "CE":
                    loss_fct = CrossEntropyLoss()
                cls_loss = loss_fct(
                    cls_logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                cls_loss = loss_fct(cls_logits, labels)

            # print("cls_loss")
            # print(cls_loss)

            # For token cls loss
            loss_fct = CrossEntropyLoss()
            token_loss = loss_fct(
                token_logits.view(-1, self.num_token_labels), token_labels.view(-1)
            )
            # print("token_loss")
            # print(token_loss)

            total_loss = cls_loss + token_loss

        if not return_dict:
            output = (cls_logits, token_logits) + outputs[2:]
            return (
                ((total_loss, cls_loss, token_loss) + output)
                if total_loss is not None
                else output
            )

        return MultiTaskBertForSequenceAndTokenClassifierOutput(
            total_loss=total_loss,
            cls_loss=cls_loss,
            token_loss=token_loss,
            cls_logits=cls_logits,
            token_logits=token_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
