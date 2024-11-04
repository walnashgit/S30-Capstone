import json
import random

import torch
from torch.utils.data import Dataset, Sampler
import h5py

from S30ProjectionTraining.util import tokenizer_image_token
from config import projection_layer_config as cfg
from config import IGNORE_INDEX


# use for direct comparison of image embedding and phi text embedding
class ProjectionLayerDataset(Dataset):
    def __init__(self, embedding_file, caption_file):
        self.embedding_file = h5py.File(embedding_file, 'r')
        self.image_ids = set(self.embedding_file.keys())
        self.caption_file = caption_file
        self.captions = self.get_caption_dict()  # List of text inputs

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        image_id = str(caption.get('image_id'))
        image_embedding = torch.tensor(self.embedding_file[image_id][()])
        text = caption.get('caption')
        # prompt = '<image> caption: '

        return image_embedding, text

    def get_caption_dict(self):
        with open(self.caption_file, 'r') as f:
            caption_file_json = json.load(f)
            # return [cap for cap in caption_file_json['annotations']]
            return [
                cap for cap in caption_file_json['annotations']
                if str(cap['image_id']) in self.image_ids
            ]


# use for combined input of image and text embedding to phi and comparison with labels
class ProjectionLayerDataset2(Dataset):
    def __init__(self, embedding_file, caption_file, tokenizer, max_length=80):
        self.embedding_file = h5py.File(embedding_file, 'r')
        self.image_ids = set(self.embedding_file.keys())
        self.caption_file = caption_file
        self.captions = self.get_caption_dict()  # List of text inputs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_token = cfg['image_token']

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        image_id = str(caption.get('image_id'))
        image_embedding = torch.tensor(self.embedding_file[image_id][()])
        image_embedding = image_embedding[1:, :]  # returning 49 x 768
        text = caption.get('caption')
        # prompt = f'{self.image_token} caption: {text}'
        prompt = f'{self.image_token} caption: '
        # tokenized_prompt = self.tokenizer(prompt, truncation=True, max_length=self.max_length, padding=True,
        #                            return_tensors=None)
        prompt_ids = torch.tensor(tokenizer_image_token(prompt, tokenizer=self.tokenizer), dtype=torch.int32)
        # prompt_ids = tokenized_prompt['input_ids']
        labels = self.tokenizer.encode(text)
        labels = self.get_final_label_ids(labels, prompt_ids, image_embedding)



        # tokenized = self.tokenizer(text, truncation=True, max_length=self.max_length, padding=True,
        #                            return_tensors=None)
        #
        # # Find the position of the <image> token
        # input_ids = tokenized['input_ids']
        # image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        # image_token_position = (torch.tensor(prompt_ids) == image_token_id).nonzero(as_tuple=True)[0]
        #
        # labels = self.get_final_token_ids(input_ids, image_embedding)

        # return image_embedding, text
        # return image_embedding, prompt

        return {
            'image_embedding': image_embedding,
            'prompt_ids': prompt_ids, #torch.tensor(prompt_ids, dtype=torch.int32),
            'labels': labels,
            # 'attention_mask': torch.tensor(tokenized['attention_mask']),
            # 'image_token_position': image_token_position
        }

    def get_final_label_ids(self, labels, prompt_ids, image_embedding):
        pad_token_count = self.max_length - (prompt_ids.size(0) + image_embedding.size(0) - 1) - len(labels) - 1
        if pad_token_count < 0:
            pad_token_count = 0
            # truncate_len = self.max_length - (len(token_ids) + prompt_ids.size(0) - 1) - 1
            truncate_len = self.max_length - (prompt_ids.size(0) + image_embedding.size(0) - 1) - 1
            labels = labels[:truncate_len]

        labels = torch.cat(
            [
                torch.tensor(labels, dtype=torch.int32),
                torch.tensor([self.tokenizer.eos_token_id], dtype=torch.int32),
                torch.tensor([self.tokenizer.pad_token_id] * pad_token_count, dtype=torch.int32)
            ],
            dim=0
        )
        return labels

    # def get_caption_dict(self):
    #     with open(self.caption_file, 'r') as f:
    #         caption_file_json = json.load(f)
    #         return [cap for cap in caption_file_json['annotations']]
    def get_caption_dict(self):
        with open(self.caption_file, 'r') as f:
            caption_file_json = json.load(f)
            # return [cap for cap in caption_file_json['annotations']]
            return [
                cap for cap in caption_file_json['annotations']
                if str(cap['image_id']) in self.image_ids
            ]


class MultiModalLlavaDataset(Dataset):
    def __init__(self, embedding_file, instruct_file, tokenizer, max_length=128):
        self.embedding_file = h5py.File(embedding_file, 'r')
        self.instruct_file = instruct_file
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_ids = set(self.embedding_file.keys())
        self.conversation_data = self.get_conversation_data()  # List of text inputs

        self.image_token = cfg['image_token2']

    def __len__(self):
        return len(self.conversation_data)

    def __getitem__(self, idx):
        conv_item = self.conversation_data[idx]
        image_id = str(conv_item.get('image_id'))

        image_embedding = torch.tensor(self.embedding_file[image_id][()])
        image_embedding = image_embedding[1:, :]  # returning 49 x 768
        conv = conv_item.get('qna')
        prompt = '<image>\n' + conv
        prompt_token_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors='pt')

        final_prompt_token_ids = self.get_final_token_ids(prompt_token_ids, image_embedding)

        labels = final_prompt_token_ids.clone()
        parts = prompt.split('AI### ')
        if len(parts) != 2:
            print(prompt)
            raise Exception("Not proper QnA text: " + conv)

        que_len = len(tokenizer_image_token(parts[0] + 'AI### ', tokenizer=self.tokenizer))
        labels[0: que_len] = IGNORE_INDEX

        return {
            'image_embedding': image_embedding,
            'input_ids': final_prompt_token_ids,
            'labels': labels
        }

    def get_final_token_ids(self, token_ids, image_embedding):
        input_pad_tokens = self.max_length - (len(token_ids) + image_embedding.size(0) - 1)

        if input_pad_tokens < 0:
            input_pad_tokens = 0
            truncate_len = self.max_length - (image_embedding.size(0) - 1)
            token_ids = token_ids[:truncate_len]

        input_ids = torch.cat(
            [
                torch.tensor(token_ids, dtype=torch.int32),
                torch.tensor([self.tokenizer.pad_token_id] * input_pad_tokens, dtype=torch.int32)
            ],
            dim=0
        )
        return input_ids

    def get_conversation_data(self):
        with open(self.instruct_file, 'r') as f:
            instruct_file_json = json.load(f)
            return self.split_conversation(instruct_file_json)

    def split_conversation(self, instruct_file_json):
        instruct_data = []
        seps = ['\n', '<|endoftext|>']
        for conv_dict in instruct_file_json:
            image_id = self.remove_leading_zeros(str(conv_dict.get('id')))
            conv = conv_dict.get('conversations')
            if image_id in self.image_ids:
                t = None
                for i, qa in enumerate(conv):
                    role = qa['from']
                    msg = qa['value'].replace('<image>', '')
                    if i % 2 == 0:
                        t = ''

                        if role == 'human':
                            t += 'Human### ' + msg + seps[0]
                    else:
                        if role == 'gpt' and t and msg:
                            t += 'AI### ' + msg + seps[1]

                        if t:
                            instruct_dict = dict(
                                image_id=image_id,
                                qna=t
                            )
                            instruct_data.append(instruct_dict)

        return instruct_data

    def remove_leading_zeros(self, number_string):
        return number_string.lstrip('0') or '0'

    def prepare_conversation(self, conversations):
        # Prepare conversation
        conversation = ""
        for turn in conversations:
            if turn['from'] == 'human':
                conversation += f"Human###: {turn['value']}\n"
            else:
                conversation += f"AI###: {turn['value']}{self.tokenizer.eos_token_id}"
        return conversation


# def bucket_and_batch(data, batch_size):
#     # Sort data by sequence length
#     sorted_data = sorted(data, key=lambda x: len(x['input_ids']))
#     batches = [sorted_data[i:i + batch_size] for i in range(0, len(sorted_data), batch_size)]
#     return batches

# Use this to create your batches before passing to DataLoader


class BucketBatchSampler(Sampler):
    def __init__(self, data_source, batch_size):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.buckets = self._create_buckets()

    def _create_buckets(self):
        # Sort data by sequence length and create buckets
        # This is a simplified example; adjust based on your actual data structure
        sorted_data = sorted(range(len(self.data_source)),
                             key=lambda idx: len(self.data_source[idx]['input_ids']))
        return [sorted_data[i:i + self.batch_size] for i in range(0, len(sorted_data), self.batch_size)]

    def __iter__(self):
        random.shuffle(self.buckets)
        for bucket in self.buckets:
            yield bucket

    def __len__(self):
        return len(self.buckets)

