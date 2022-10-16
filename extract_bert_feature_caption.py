import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").cuda()

with open('data/ground_results_vg_all_v4.json', 'r') as f:
    data = json.load(f)

empty_num = 0
for image_data in tqdm(data, ncols=100, ascii=True):
    if len(image_data['captions']) == 0:
        empty_num += 1
        continue

    image_id = image_data['image_id']

    # tokenize all captions of this image
    inputs_ids = tokenizer(image_data['captions'])['input_ids']
    inputs_t = {
        k: v.cuda() for k, v in tokenizer(
            image_data['captions'],
            return_tensors="pt",
            padding='longest'
        ).items()
    }

    # convert to token strings for matching
    tokens = [tokenizer.convert_ids_to_tokens(sent) for sent in inputs_ids]

    # for each caption, match scene graph entities to bert tokens
    for cap_scene, cap_tokens in zip(image_data['scene_graphs'], tokens):
        i = 0  # match from the start
        for ent in cap_scene['entities']:
            ent_head = ent['head']
            # from i-th token, try finding current entity
            while i < len(cap_tokens):
                # find a full word from i-th token
                full_word = cap_tokens[i]
                j = i + 1
                while j < len(cap_tokens) and cap_tokens[j][:2] == '##':
                    full_word += cap_tokens[j][2:]
                    j += 1
                # find the word or not
                if full_word == ent_head:
                    ent['span'] = [i, j]
                    i = j
                    break
                else:
                    i = j
            if 'span' not in ent:
                raise RuntimeError('entity <{}> not in {}.'.format(ent_head, cap_tokens))

    features = model(**inputs_t)['last_hidden_state']
    feat_list = []
    for cap_idx, cap_scene in enumerate(image_data['scene_graphs']):
        for ent in cap_scene['entities']:
            feat_list.append(features[cap_idx, slice(*ent['span'])].mean(dim=0, keepdim=True).to('cpu'))
    ent_feat = torch.cat(feat_list, dim=0)
    torch.save(ent_feat, 'cache/ent_bert_feat/{}.pt'.format(image_id))

print('empty:', empty_num)

# merge bert label
with open('cache/split.json', 'r') as f:
    splits = json.load(f)

for split in ['train', 'val']:
    bert_feat_list = []
    for image_id in tqdm(splits[split], ncols=100, desc='merging {}'.format(split)):
        ent_bert_feat = torch.load('cache/ent_bert_feat/{}.pt'.format(image_id))
        bert_feat_list.append(ent_bert_feat)
    merged_bert_feat = torch.cat(bert_feat_list, dim=0)
    torch.save(merged_bert_feat, 'cache/{}_bert_feat.pt'.format(split))
