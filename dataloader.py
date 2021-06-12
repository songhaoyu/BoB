# Copyright 2021 Haoyu Song
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import json

'''
DATASETS

    ConvAI2 PersonaChat:
        language:           English
        format:             personal facts
        persona type:       dense
        downloading url:    http://parl.ai/downloads/convai2/convai2_fix_723.tgz
        data to use:        train_self_original_no_cands & valid_self_original_no_cands

    ECDT2019 PersonalDialog:
        language:           Chinese
        format:             profiles
        persona type:       sparse
        data to use:        dialogues_train.json & test_data_random.json
'''


class ConvAI2Dataset(torch.utils.data.Dataset):
    def __init__(self, persona, queries, labels, device):
        self.persona = persona
        self.queries = queries
        self.labels = labels
        self.device = device

    def __getitem__(self, idx):
        persona = {
            key: torch.tensor(val[idx]).to(self.device)
            for key, val in self.persona.items()
        }
        query = {
            key: torch.tensor(val[idx]).to(self.device)
            for key, val in self.queries.items()
        }
        response = {
            key: torch.tensor(val[idx]).to(self.device)
            for key, val in self.labels.items()
        }
        return {'persona': persona, 'query': query, 'response': response}

    def __len__(self):
        return len(self.labels['input_ids'])


class ECDT2019Dataset(torch.utils.data.Dataset):
    def __init__(self, profiles, queries, responses, device):
        self.profiles = profiles
        self.queries = queries
        self.responses = responses
        self.device = device

    def __getitem__(self, idx):
        profile = {
            key: torch.tensor(val[idx]).to(self.device)
            for key, val in self.profiles.items()
        }
        query = {
            key: torch.tensor(val[idx]).to(self.device)
            for key, val in self.queries.items()
        }
        response = {
            key: torch.tensor(val[idx]).to(self.device)
            for key, val in self.responses.items()
        }
        return {'persona': profile, 'query': query, 'response': response}

    def __len__(self):
        return len(self.responses['input_ids'])


class NLIDataset(torch.utils.data.Dataset):
    def __init__(self, pre, hyp, device):
        self.pre = pre
        self.hyp = hyp
        self.device = device

    def __getitem__(self, idx):
        pre = {
            key: torch.tensor(val[idx]).to(self.device)
            for key, val in self.pre.items()
        }
        hyp = {
            key: torch.tensor(val[idx]).to(self.device)
            for key, val in self.hyp.items()
        }
        return {'pre': pre, 'hyp': hyp}

    def __len__(self):
        return len(self.pre['input_ids'])




def read_convai2_split(split_dir):
    persona = []
    query = []
    response = []
    try:
        with open(split_dir, "r", encoding="utf-8") as src:
            pre_st, st = 'dia', 'dia'
            for line in src:
                line = line.strip()
                if 'your persona:' in line:
                    pre_st = st
                    st = 'per'
                else:
                    pre_st = st
                    st = 'dia'
                if pre_st == 'dia' and st == 'per':
                    per_group = ''
                if st == 'per':
                    per_group+=(line[16:]+' ')
                elif st == 'dia':
                    persona.append(per_group)
                    line = line[line.find(' '):]
                    query.append(line.split('\t')[0])
                    response.append(line.split('\t')[1])
                else:
                    raise (ValueError)
    except FileNotFoundError:
        print(f"Sorry! The file {split_dir} can't be found.")
    return persona, query, response


def read_ecdt2019_split(split_dir, split_type='train'):
    profile_lst = []
    query_lst = []
    response_lst = []
    try:
        with open(split_dir, "r", encoding="utf-8") as src:
            for line in src:
                line = line.strip()
                data_dict = json.loads(line)

                if split_type == 'test':
                    gr = data_dict['golden_response'][0]
                    response_lst.append(''.join(gr.split(' ')))

                    dialog = data_dict['dialog']
                    uid = data_dict['uid']
                    profile = data_dict['profile']
                    
                    q = dialog[-1][0]
                    pfl = str(profile[uid[-1]])
                    query_lst.append(''.join(q.split(' ')))
                    profile_lst.append(pfl)
                
                elif split_type == 'train':
                    dialog = data_dict['dialog']
                    uid = data_dict['uid']
                    profile = data_dict['profile']
                    
                    q, r = dialog[-2][0], dialog[-1][0]
                    pfl = str(profile[uid[-1]])
                    query_lst.append(''.join(q.split(' ')))
                    response_lst.append(''.join(r.split(' ')))
                    profile_lst.append(pfl)

                else:
                    print(f'Invalid split_type {split_type}.')
                    raise(ValueError)

    except FileNotFoundError:
        print(f"Sorry! The file {split_dir} can't be found.")
    return profile_lst, query_lst, response_lst


def read_nli_split(split_dir):
    pre_lst = []
    hyp_lst = []
    try:
        with open(split_dir, "r", encoding="utf-8") as src:
            for line in src:
                line = line.strip()
                sent_1, sent_2 = line.split('\t')[0], line.split('\t')[1]
                if len(sent_1.split(' ')) > len(sent_2.split(' ')):
                    pre, hyp = sent_1, sent_2
                else:
                    pre, hyp = sent_2, sent_1
                pre_lst.append(pre)
                hyp_lst.append(hyp)
    except FileNotFoundError:
        print(f"Sorry! The file {split_dir} can't be found.")
    return pre_lst, hyp_lst