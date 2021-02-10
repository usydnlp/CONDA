import os
import copy
import json
import logging

import torch
from torch.utils.data import TensorDataset

from utils import get_intent_labels, get_slot_labels

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        intent_label: (Optional) string. The intent label of the example.
        slot_labels: (Optional) list. The slot labels of the example.
    """

    def __init__(self, guid, words, intent_label=None, slot_labels=None):
        self.guid = guid
        self.words = words
        self.intent_label = intent_label
        self.slot_labels = slot_labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, intent_label_id, slot_labels_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.intent_label_id = intent_label_id
        self.slot_labels_ids = slot_labels_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class JointProcessor(object):
    """Processor for the JointBERT data set """

    def __init__(self, args):
        self.args = args
        self.intent_labels = get_intent_labels(args)
        self.slot_labels = get_slot_labels(args)

        self.input_text_file = 'seq.in'
        self.intent_label_file = 'label'
        self.slot_labels_file = 'seq.out'

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, texts, intents, slots, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, (text, intent, slot) in enumerate(zip(texts, intents, slots)):
            guid = "%s-%s" % (set_type, i)
            # 1. input_text
            words = text.split()  # Some are spaced twice
            # 2. intent
            intent_label = self.intent_labels.index(intent) if intent in self.intent_labels else self.intent_labels.index("UNK")
            # 3. slot
            slot_labels = []
            for s in slot.split():
                slot_labels.append(self.slot_labels.index(s) if s in self.slot_labels else self.slot_labels.index("UNK"))

            # assert len(words) == len(slot_labels)
            examples.append(InputExample(guid=guid, words=words, intent_label=intent_label, slot_labels=slot_labels))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        data_path = os.path.join(self.args.data_dir, self.args.task, mode)
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_examples(texts=self._read_file(os.path.join(data_path, self.input_text_file)),
                                     intents=self._read_file(os.path.join(data_path, self.intent_label_file)),
                                     slots=self._read_file(os.path.join(data_path, self.slot_labels_file)),
                                     set_type=mode)


processors = {
    "atis": JointProcessor,
    "snips": JointProcessor,
    "dataset": JointProcessor,
    "standard_dota50k": JointProcessor,
    'standard_dota50k_context_depth=3':JointProcessor,
    'standard_dota50k_context_depth=all':JointProcessor,
    '45k_context_depth==2':JointProcessor,
    '45k_context_depth==4':JointProcessor,
    '45k_context_depth==6':JointProcessor,
    '45k_context_depth==ALL':JointProcessor,
}


def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 pad_token_label_id=-100,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        # print(example)
        if ex_index % 5000 == 100:
            # print(1/0)
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        all_context_words = []
        single_utterance = []
        for ind, word in enumerate(example.words):
          if word == '[eos]' or ind == len(example.words)-1:
            if word == '[eos]':
              all_context_words.append(single_utterance)
              single_utterance = []
            else:
              single_utterance.append(word)
              all_context_words.append(single_utterance)
              single_utterance = []
          else:
            single_utterance.append(word)
        # print(all_context_words)
        # Tokenize word by word (for NER)
        # do_slot = False
        do_slot = True
        input_ids_all = []
        attention_mask_all = []
        token_type_ids_all = []

        for ind, utterance in enumerate(all_context_words):
          # utterance = all_context_words[-1]
          # print('utterance:',utterance)
          # if ind == len(all_context_words) -1:
          #   # print('utterance:',utterance)
          #   do_slot = True
          tokens = []
          slot_labels_ids = []
          for word, slot_label in zip(utterance, example.slot_labels):
              word_tokens = tokenizer.tokenize(word)
              if not word_tokens:
                  word_tokens = [unk_token]  # For handling the bad-encoded word
              tokens.extend(word_tokens)
              # Use the real label id for the first token of the word, and padding ids for the remaining tokens
              if do_slot:
                slot_labels_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))
          # print('tokens:',tokens)
          # Account for [CLS] and [SEP]
          special_tokens_count = 2
          if len(tokens) > max_seq_len - special_tokens_count:
              tokens = tokens[:(max_seq_len - special_tokens_count)]
              if do_slot:
                slot_labels_ids = slot_labels_ids[:(max_seq_len - special_tokens_count)]

          # Add [SEP] token
          tokens += [sep_token]
          if do_slot:
            slot_labels_ids += [pad_token_label_id]
          token_type_ids = [sequence_a_segment_id] * len(tokens)

          # Add [CLS] token
          tokens = [cls_token] + tokens
          if do_slot:
            slot_labels_ids = [pad_token_label_id] + slot_labels_ids
          token_type_ids = [cls_token_segment_id] + token_type_ids

          input_ids = tokenizer.convert_tokens_to_ids(tokens)

          # The mask has 1 for real tokens and 0 for padding tokens. Only real
          # tokens are attended to.
          attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

          # Zero-pad up to the sequence length.
          padding_length = max_seq_len - len(input_ids)
          input_ids = input_ids + ([pad_token_id] * padding_length)
          attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
          token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
          if do_slot:
            slot_labels_ids = slot_labels_ids + ([pad_token_label_id] * padding_length)

          assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
          assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
          assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)
          if do_slot:
            # print('do slot')
            assert len(slot_labels_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(len(slot_labels_ids), max_seq_len)
          # print('input_ids:',len(input_ids))
          # print('attention_mask:',len(attention_mask))
          # print('token_type_ids:',len(token_type_ids))
          input_ids_all.extend(input_ids)
          attention_mask_all.extend(attention_mask)
          token_type_ids_all.extend(token_type_ids)
          # break
        # print(len(input_ids_all))
        # print(len(attention_mask_all))
        # print(len(token_type_ids_all))

        intent_label_id = int(example.intent_label)
        # print('intent_label_id:',intent_label_id)
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("intent_label: %s (id = %d)" % (example.intent_label, intent_label_id))
            logger.info("slot_labels: %s" % " ".join([str(x) for x in slot_labels_ids]))

        features.append(
            InputFeatures(input_ids=input_ids_all,
                          attention_mask=attention_mask_all,
                          token_type_ids=token_type_ids_all,
                          intent_label_id=intent_label_id,
                          slot_labels_ids=slot_labels_ids
                          ))
    # for (ex_index, example) in enumerate(examples):
    #     # print(example)
    #     if '[eos]' in example.words:
    #       words = example.words
    #       words.reverse()
    #       index_e = words.index('[eos]')
    #       words.reverse()
    #       example.words = words[-index_e:] 
    #     # print(example.words)
    #     if ex_index % 5000 == 100:
    #         # print(1/0)
    #         logger.info("Writing example %d of %d" % (ex_index, len(examples)))

    #     # Tokenize word by word (for NER)
    #     tokens = []
    #     slot_labels_ids = []
    #     for word, slot_label in zip(example.words, example.slot_labels):
    #         word_tokens = tokenizer.tokenize(word)
    #         if not word_tokens:
    #             word_tokens = [unk_token]  # For handling the bad-encoded word
    #         tokens.extend(word_tokens)
    #         # Use the real label id for the first token of the word, and padding ids for the remaining tokens
    #         slot_labels_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))

    #     # Account for [CLS] and [SEP]
    #     special_tokens_count = 2
    #     if len(tokens) > max_seq_len - special_tokens_count:
    #         tokens = tokens[:(max_seq_len - special_tokens_count)]
    #         slot_labels_ids = slot_labels_ids[:(max_seq_len - special_tokens_count)]

    #     # Add [SEP] token
    #     tokens += [sep_token]
    #     slot_labels_ids += [pad_token_label_id]
    #     token_type_ids = [sequence_a_segment_id] * len(tokens)

    #     # Add [CLS] token
    #     tokens = [cls_token] + tokens
    #     slot_labels_ids = [pad_token_label_id] + slot_labels_ids
    #     token_type_ids = [cls_token_segment_id] + token_type_ids

    #     input_ids = tokenizer.convert_tokens_to_ids(tokens)

    #     # The mask has 1 for real tokens and 0 for padding tokens. Only real
    #     # tokens are attended to.
    #     attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    #     # Zero-pad up to the sequence length.
    #     padding_length = max_seq_len - len(input_ids)
    #     input_ids = input_ids + ([pad_token_id] * padding_length)
    #     attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
    #     token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
    #     slot_labels_ids = slot_labels_ids + ([pad_token_label_id] * padding_length)

    #     assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
    #     assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
    #     assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)
    #     assert len(slot_labels_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(len(slot_labels_ids), max_seq_len)

    #     intent_label_id = int(example.intent_label)

    #     if ex_index < 5:
    #         logger.info("*** Example ***")
    #         logger.info("guid: %s" % example.guid)
    #         logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
    #         logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    #         logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
    #         logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
    #         logger.info("intent_label: %s (id = %d)" % (example.intent_label, intent_label_id))
    #         logger.info("slot_labels: %s" % " ".join([str(x) for x in slot_labels_ids]))

    #     features.append(
    #         InputFeatures(input_ids=input_ids,
    #                       attention_mask=attention_mask,
    #                       token_type_ids=token_type_ids,
    #                       intent_label_id=intent_label_id,
    #                       slot_labels_ids=slot_labels_ids
    #                       ))

    return features


def load_and_cache_examples(args, tokenizer, mode):
    processor = processors[args.task](args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}_{}'.format(
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len
        )
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        pad_token_label_id = args.ignore_index
        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer,
                                                pad_token_label_id=pad_token_label_id)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    # all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    # all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    # all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    # all_intent_label_ids = torch.tensor([f.intent_label_id for f in features], dtype=torch.long)
    # all_slot_labels_ids = torch.tensor([f.slot_labels_ids for f in features], dtype=torch.long)

    # dataset = TensorDataset(all_input_ids, all_attention_mask,
    #                         all_token_type_ids, all_intent_label_ids, all_slot_labels_ids)
    all_input_ids = [f.input_ids for f in features]
    all_attention_mask = [f.attention_mask for f in features]
    all_token_type_ids = [f.token_type_ids for f in features]
    all_intent_label_ids = [f.intent_label_id for f in features]
    all_slot_labels_ids = [f.slot_labels_ids for f in features]

    dataset = CustomerDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_intent_label_ids, all_slot_labels_ids)

    return dataset

from torch.utils.data import Dataset
class CustomerDataset(Dataset):   
    def __init__(self,all_input_ids,all_attention_mask,all_token_type_ids,all_intent_label_ids,all_slot_labels_ids): 

        self.all_input_ids = all_input_ids
        self.all_attention_mask = all_attention_mask
        self.all_token_type_ids = all_token_type_ids
        self.all_intent_label_ids = all_intent_label_ids
        self.all_slot_labels_ids = all_slot_labels_ids

    def __len__(self):  
        return len(self.all_input_ids)

    def __getitem__(self,idx):

        input_id = torch.tensor(self.all_input_ids[idx],dtype=torch.long)
        attention_mask = torch.tensor(self.all_attention_mask[idx],dtype=torch.long)
        token_type_ids = torch.tensor(self.all_token_type_ids[idx],dtype=torch.long)
        intent_label_ids = torch.tensor(self.all_intent_label_ids[idx],dtype=torch.long)
        slot_labels_ids = torch.tensor(self.all_slot_labels_ids[idx],dtype=torch.long)

        sample = {'input_id':input_id, 'attention_mask':attention_mask,'token_type_ids':token_type_ids,'intent_label_ids':intent_label_ids,'slot_labels_ids':slot_labels_ids,}


        return sample
