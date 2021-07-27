import torch
import torch.nn as nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertConfig
from torchcrf import CRF
from .module import IntentClassifier, SlotClassifier


class JointBERT(BertPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super(JointBERT, self).__init__(config)
        self.args = args
        self.do_separate = args.do_separate
        self._device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.bert = BertModel(config=config)  # Load pretrained bert
        if self.do_separate!=0:
          self.intent_classifier_for_one_utterance = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)
        self.intent_classifier = IntentClassifier(config.hidden_size+500, self.num_intent_labels, args.dropout_rate)
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, args.dropout_rate)

        self.context_lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=500, num_layers=1, batch_first=True)

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids):
        # print('input_ids:',input_ids.shape)
        num_of_utterance = int(input_ids.shape[1]/self.args.max_seq_len)
        concat_lstm_input = None
        for i in range(num_of_utterance-1):
          input_ids_single_utterance = input_ids[:,i*100:(i+1)*100]
          attention_mask_single_utterance = attention_mask[:,i*100:(i+1)*100]
          token_type_id_single_utterance = token_type_ids[:,i*100:(i+1)*100]
          outputs = self.bert(input_ids_single_utterance, attention_mask=attention_mask_single_utterance,
                              token_type_ids=token_type_id_single_utterance)  # sequence_output, pooled_output, (hidden_states), (attentions)
          sequence_output = outputs[0]
          pooled_output = outputs[1]  # [CLS]
          if concat_lstm_input == None:
            concat_lstm_input = pooled_output.view(1,1,-1)
          else:
            concat_lstm_input = torch.cat((concat_lstm_input,pooled_output.view(1,1,-1)),dim=1)

        if concat_lstm_input!=None:
          lstm_output, (h_n, c_n) = self.context_lstm(concat_lstm_input)
          lstm_output = lstm_output[:,-1,:].view(1,-1)

        input_id_last_utterance = input_ids[:,-100:]
        attention_mask_last_utterance = attention_mask[:,-100:]
        token_type_id_last_utterance = token_type_ids[:,-100:]
        outputs = self.bert(input_id_last_utterance, attention_mask=attention_mask_last_utterance,
                              token_type_ids=token_type_id_last_utterance)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        if concat_lstm_input!=None:
          pooled_output = torch.cat((lstm_output,pooled_output),dim=1)
        else:
          if self.do_separate==0:
            lstm_output = torch.zeros(1, 500).to(self._device)
            pooled_output = torch.cat((lstm_output,pooled_output),dim=1)

        if self.do_separate!=0 and num_of_utterance == 1:
          intent_logits = self.intent_classifier_for_one_utterance(pooled_output)
        else:
          intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += intent_loss

        # 2. Slot Softmax
        # if slot_labels_ids is not None:
        #     if self.args.use_crf:
        #         slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
        #         slot_loss = -1 * slot_loss  # negative log-likelihood
        #     else:
        #         slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
        #         # Only keep active parts of the loss
        #         if attention_mask is not None:
        #             active_loss = attention_mask.view(-1) == 1
        #             active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
        #             active_labels = slot_labels_ids.view(-1)[active_loss]
        #             slot_loss = slot_loss_fct(active_logits, active_labels)
        #         else:
        #             slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
        #     total_loss += self.args.slot_loss_coef * slot_loss

        outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits
