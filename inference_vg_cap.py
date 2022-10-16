import json
import pickle
import os
from collections import defaultdict
from argparse import ArgumentParser

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from grounder import InferenceDataset, DropoutGrounderV3


def grounder_inference(grounder, ent_feat, box_feat, valid_mask):
    with torch.no_grad():
        ent_feat = ent_feat.to('cuda:0')
        box_feat = box_feat.to('cuda:0')
        valid_mask = valid_mask.to('cuda:0', dtype=torch.float)
        score, _, _ = grounder(ent_feat, box_feat)
        score = valid_mask * score + (1 - valid_mask) * (-1e7)
        score = score.to('cpu')
        prob = torch.nn.functional.softmax(score, dim=1)
        pred = torch.argsort(prob, dim=1, descending=True)[:, :5]
        prob = prob[torch.arange(len(score)).unsqueeze(dim=1).repeat_interleave(5, dim=1), pred]
    return pred, prob


def main():
    parser = ArgumentParser()
    parser.add_argument('--m', required=True, help='model timestamp')
    args = parser.parse_args()

    model_dir = 'output/{}'.format(args.m)
    with open(os.path.join(model_dir, 'config.json'), 'r') as f:
        cfg = json.load(f)
    grounder = DropoutGrounderV3(dropout_p=cfg['config']['DROPOUT'])
    grounder.load_state_dict(torch.load(os.path.join(model_dir, 'grounder_best.pth')))

    trn_dataset = InferenceDataset(split='train')
    val_dataset = InferenceDataset(split='val')
    trn_loader = DataLoader(trn_dataset, batch_size=1024, shuffle=False, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=16)

    grounder.to('cuda:0')
    grounder.eval()

    pred_list, prob_list = [], []
    with tqdm(total=len(trn_dataset), ncols=120, desc='inference train', ascii=True) as pbar:
        for batch in trn_loader:
            pred, prob = grounder_inference(grounder=grounder, **batch)  # [N, 5]
            pred_list.append(pred)
            prob_list.append(prob)
            pbar.update(batch['valid_mask'].size(0))
    with tqdm(total=len(val_dataset), ncols=120, desc='inference val', ascii=True) as pbar:
        for batch in val_loader:
            pred, prob = grounder_inference(grounder=grounder, **batch)  # [N, 5]
            pred_list.append(pred)
            prob_list.append(prob)
            pbar.update(batch['valid_mask'].size(0))
    all_pred = torch.cat(pred_list, dim=0).tolist()
    all_prob = torch.cat(prob_list, dim=0).tolist()

    image_id_list = []
    with open('cache/train_image_id_list.json', 'r') as f:
        image_id_list.extend(json.load(f))
    with open('cache/val_image_id_list.json', 'r') as f:
        image_id_list.extend(json.load(f))

    with open('data/process_results_vg_all_v4.pkl', 'rb') as f:
        gd = pickle.load(f)

    assert (
        len(all_pred) == len(all_prob) == len(image_id_list)
        == sum(len(v['gd_boxes']) for v in gd.values())
    )

    pred_dict = defaultdict(list)
    prob_dict = defaultdict(list)
    for pred, prob, image_id in zip(all_pred, all_prob, image_id_list):
        pred_dict[image_id].append(pred)
        prob_dict[image_id].append(prob)

    # Fill in new results
    for image_id in pred_dict.keys():
        gd[image_id]['gd_boxes'] = pred_dict[image_id]
        gd[image_id]['gd_scores'] = prob_dict[image_id]

    print('saving grounding results to {}'.format(model_dir))
    with open(os.path.join(model_dir, 'vg_all_v4_grounder.pkl'), 'wb') as f:
        pickle.dump(gd, f)


if __name__ == '__main__':
    main()
