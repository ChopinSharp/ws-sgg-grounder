import json
import torch
import torch.nn as nn
from torch.nn.functional import normalize
from torch.utils.data import Dataset
import numpy as np
import numpy.random as npr
import random


class AdaptiveSmoothLabelEntityGroundingDataset(Dataset):

    def __init__(self, split='val', truncate=True, smooth_ratio=0.1):
        super(AdaptiveSmoothLabelEntityGroundingDataset, self).__init__()
        self.split = split
        self.smooth_ratio = smooth_ratio
        with open('cache/{}_image_id_list.json'.format(split), 'r') as f:
            image_id_list = json.load(f)
        bert_feat = torch.load('cache/{}_bert_feat.pt'.format(split))
        bert_feat.requires_grad_(False)
        tag_label = torch.load('cache/{}_soft_tag_label.pt'.format(split))
        albef_label = torch.load('cache/{}_soft_albef_label.pt'.format(split))
        valid_mask = torch.load('cache/{}_valid_mask.pt'.format(split))
        self.bert_feat = bert_feat
        self.valid_mask = valid_mask
        self.label_1 = nn.functional.softmax(tag_label, dim=1)
        self.label_2 = nn.functional.softmax(albef_label, dim=1)
        self.image_id_list = image_id_list
        # Load hard label for recall evaluation
        hard_tag_label = torch.load('cache/{}_tag_label.pt'.format(split))
        hard_albef_label = torch.load('cache/{}_albef_label.pt'.format(split))
        bg_mask = torch.load('cache/{}_bg_mask.pt'.format(split))
        self.hard_label = (1 - (1 - hard_tag_label) * (1 - bg_mask)) * hard_albef_label
        # Replace tag label with ALBEF label
        replace_mask = (hard_tag_label.sum(dim=1) == 0)
        self.label_1[replace_mask] = self.label_2[replace_mask]

        self.hard_tag_label = hard_tag_label
        self.hard_albef_label = hard_albef_label
        self.hard_tag_label[replace_mask] = self.hard_albef_label[replace_mask]

        # Filter out samples without hard fg
        filter_mask = self.hard_label.sum(dim=1) > 0
        self.bert_feat = self.bert_feat[filter_mask]
        self.valid_mask = self.valid_mask[filter_mask]
        self.label_1 = self.label_1[filter_mask]
        self.label_2 = self.label_2[filter_mask]
        self.hard_label = self.hard_label[filter_mask]
        self.image_id_list = torch.tensor(image_id_list)[filter_mask].tolist()

        self.hard_tag_label = self.hard_tag_label[filter_mask]
        self.hard_albef_label = self.hard_albef_label[filter_mask]

        smooth_mask_1 = torch.sum(self.label_1 > 0, dim=1) < 36
        self.label_1[smooth_mask_1] = self._labelSmooth(self.label_1[smooth_mask_1], epsilon=smooth_ratio)
        smooth_mask_2 = torch.sum(self.label_2 > 0, dim=1) < 36
        self.label_2[smooth_mask_2] = self._labelSmooth(self.label_2[smooth_mask_2], epsilon=smooth_ratio)
        assert torch.sum(torch.isnan(self.label_1)) == 0, 'NaN in TAG label ({})'.format(torch.sum(torch.isnan(self.label_1)))
        assert torch.sum(torch.isnan(self.label_2)) == 0, 'NaN in TAG label ({})'.format(torch.sum(torch.isnan(self.label_2)))
        # Cut val split
        if split == 'val' and truncate:
            self.image_id_list = self.image_id_list[:102400]
        print('AdaptiveSmoothLabelEntityGroundingDataset {} split done loading: {} samples.'.format(split, len(self.image_id_list)))
        self.image_id_pool = list(set(self.image_id_list))

    def __getitem__(self, index):
        image_id = self.image_id_list[index]
        box_feat = np.load(
            'data/VG_detection_results_oid/_att/{}.npz'.format(image_id),
            allow_pickle=True, encoding='latin1'
        )['feat']
        padded_box_feat = torch.zeros([36, 1536], dtype=torch.float)
        box_feat = torch.tensor(box_feat)           # [*, 1536]
        padded_box_feat[:len(box_feat)] = box_feat  # [36, 1536]

        neg_label_1 = self._generate_neg_labels_v3(self.label_1[index], 1)
        neg_label_2 = self._generate_neg_labels_v3(self.label_2[index], 1)

        return dict(
            ent_feat=self.bert_feat[index],
            box_feat=padded_box_feat,
            label_1=self.label_1[index],
            label_2=self.label_2[index],
            valid_mask=self.valid_mask[index],
            hard_label=self.hard_label[index],
            neg_label_1=neg_label_1.detach(),
            neg_label_2=neg_label_2.detach()
        )

    def _labelSmooth(self, label, epsilon=0.1):
        mask = (label > 0).float()
        len_gt = torch.sum(mask, dim=-1, keepdim=True)
        s_label = mask * (label*(1-epsilon)) + (1-mask)*epsilon/(36-len_gt)

        return s_label

    def _generate_neg_labels_v3(self, label, num_neg=1):
        num_neg_ = 10
        # neg_label = torch.zeros([num_neg, label.shape[0]], dtype=torch.float)
        neg_label = torch.zeros_like(label)
        neg_idx = label.argsort()[:num_neg_]
        num_v = torch.randperm(5)[0]
        neg_idx = neg_idx[torch.randperm(neg_idx.shape[0])[:num_v]]
        neg_label[neg_idx] = 1 / num_v
        return neg_label.view(1, -1)

    def __len__(self):
        return len(self.image_id_list)


class MILEntityGroundingDatasetV2(Dataset):

    def __init__(self, split='val', truncate=True, smooth_ratio=0.1, ent_num=10, ent_thresh=6):
        super(MILEntityGroundingDatasetV2, self).__init__()
        self.split = split
        with open('cache/split.json', 'r') as f:
            self.image_id_list = json.load(f)[split]
        self.smooth_ratio = smooth_ratio
        self.ent_num = ent_num
        # Filter trian split
        self.image_id_list = [
            image_id for image_id in self.image_id_list
            if len(torch.load('cache/ent_bert_feat/{}.pt'.format(image_id))) >= ent_thresh
        ]
        # Cut val split
        if split == 'val' and truncate:
            self.image_id_list = self.image_id_list[:10240]
        print('MILEntityGroundingDataset {} split done loading: {} samples.'.format(split.ljust(5), len(self.image_id_list)))

    def __getitem__(self, index):
        image_id = self.image_id_list[index]
        # Load labels
        hard_tag_label = torch.load('cache/ent_tag_label/{}.pt'.format(image_id))
        hard_albef_label = torch.load('cache/ent_albef_label/{}.pt'.format(image_id))
        bg_mask = torch.load('cache/bg_mask/{}.pt'.format(image_id))
        self.hard_label = (1 - (1 - hard_tag_label) * (1 - bg_mask)) * hard_albef_label

        # Load box feature
        box_feat = np.load(
            'data/VG_detection_results_oid/_att/{}.npz'.format(image_id),
            allow_pickle=True, encoding='latin1'
        )['feat']
        padded_box_feat = torch.zeros([36, 1536], dtype=torch.float)
        box_feat = torch.tensor(box_feat)           # [*, 1536]
        padded_box_feat[:len(box_feat)] = box_feat  # [36, 1536]
        valid_mask = torch.zeros(36, dtype=torch.float)
        valid_mask[:len(box_feat)] = 1.0
        # Load entity feature
        ent_feat = torch.load('cache/ent_bert_feat/{}.pt'.format(image_id))
        ent_feat.requires_grad_(False)
        # Sampling with padding
        sample_idx = None
        if len(ent_feat) >= self.ent_num:
            sample_idx = npr.choice(len(ent_feat), self.ent_num, replace=False)
        else:
            sample_idx = npr.choice(len(ent_feat), self.ent_num, replace=True)

        return dict(
            ent_feat=ent_feat[sample_idx],
            box_feat=padded_box_feat,            # [*, 36, 1536]
            valid_mask=valid_mask,
            hard_label=self.hard_label[sample_idx],
        )

    def __len__(self):
        return len(self.image_id_list)


class InferenceDataset(Dataset):

    def __init__(self, split='val'):
        super(InferenceDataset, self).__init__()
        self.split = split
        with open('cache/{}_image_id_list.json'.format(split), 'r') as f:
            image_id_list = json.load(f)
        bert_feat = torch.load('cache/{}_bert_feat.pt'.format(split))
        bert_feat.requires_grad_(False)
        valid_mask = torch.load('cache/{}_valid_mask.pt'.format(split))
        self.bert_feat = bert_feat
        self.valid_mask = valid_mask
        self.image_id_list = image_id_list
        assert len(self.image_id_list) == len(self.valid_mask) == len(self.bert_feat)

    def __getitem__(self, index):
        image_id = self.image_id_list[index]
        box_feat = np.load(
            'data/VG_detection_results_oid/_att/{}.npz'.format(image_id),
            allow_pickle=True, encoding='latin1'
        )['feat']
        padded_box_feat = torch.zeros([36, 1536], dtype=torch.float)
        box_feat = torch.tensor(box_feat)           # [*, 1536]
        padded_box_feat[:len(box_feat)] = box_feat  # [36, 1536]
        return dict(
            ent_feat=self.bert_feat[index],
            box_feat=padded_box_feat,
            valid_mask=self.valid_mask[index],
        )

    def __len__(self):
        return len(self.image_id_list)


class DropoutGrounderV3(nn.Module):

    def __init__(self, dropout_p):
        super(DropoutGrounderV3, self).__init__()
        self.visual_encoder = nn.Sequential(
            nn.Linear(1536, 768, bias=True),
            nn.ReLU(),
            nn.Linear(768, 768, bias=True)
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(768, 768, bias=True),
            nn.ReLU(),
            nn.Linear(768, 768, bias=True)
        )
        self.classifier = nn.Linear(768, 1, bias=True)
        self.adaptor = nn.Linear(768, 1, bias=True)
        self.XE_loss = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, ent_feat, box_feat, pos_att_1=None, neg_att_1=None, pos_att_2=None, neg_att_2=None):
        """

        Args:
            ent_feat: [N, 768]
            box_feat: [N, *, 1536]
            pos_att:  [N, 36]
            neg_att:  [N, 5, 36]

        Returns:
            score: [N, *]

        """
        ent_feat = self.text_encoder(ent_feat)           # [N, 768]
        box_feat = self.visual_encoder(box_feat)         # [N, 36, 768]
        ent_feat_d = ent_feat.detach()                   # [N, 768]
        box_feat_d = box_feat.detach()                   # [N, 36, 768]
        ent_feat = ent_feat.unsqueeze(dim=1)             # [N, 1, 768]
        ent_feat = normalize(ent_feat, dim=2)            # [N, 1, 768]
        box_feat = normalize(box_feat, dim=2)            # [N, 36, 768]
        merged_feat = self.dropout(ent_feat * box_feat)  # [N, 36, 768]
        score = self.classifier(merged_feat)             # [N, 36, 1]
        score = score.squeeze(dim=2)                     # [N, 36]

        if pos_att_1 is not None:
            pos_attended_box_feat_1 = torch.sum(pos_att_1.unsqueeze(dim=2) * box_feat_d, dim=1)                   # [N, 768]
            pos_attended_box_feat_2 = torch.sum(pos_att_2.unsqueeze(dim=2) * box_feat_d, dim=1)                   # [N, 768]
            pos_attended_box_feat_1 = nn.functional.normalize(pos_attended_box_feat_1, dim=1)                     # [N, 768]
            pos_attended_box_feat_2 = nn.functional.normalize(pos_attended_box_feat_2, dim=1)                     # [N, 768]
            pos_weight_1 = self.adaptor(pos_attended_box_feat_1 * ent_feat_d)                                     # [N, 1]
            pos_weight_2 = self.adaptor(pos_attended_box_feat_2 * ent_feat_d)                                     # [N, 1]

            """adaptive2"""
            neg_attended_box_feat_1 = torch.sum(neg_att_1.unsqueeze(dim=3) * box_feat_d.unsqueeze(dim=1), dim=2)  # [N, 5, 768]
            neg_attended_box_feat_2 = torch.sum(neg_att_2.unsqueeze(dim=3) * box_feat_d.unsqueeze(dim=1), dim=2)  # [N, 5, 768]
            neg_attended_box_feat_1 = nn.functional.normalize(neg_attended_box_feat_1, dim=2)                     # [N, 5, 768]
            neg_attended_box_feat_2 = nn.functional.normalize(neg_attended_box_feat_2, dim=2)                     # [N, 5, 768]
            neg_weight_1 = self.adaptor(neg_attended_box_feat_1 * ent_feat_d.unsqueeze(dim=1))                    # [N, 5, 1]
            neg_weight_2 = self.adaptor(neg_attended_box_feat_2 * ent_feat_d.unsqueeze(dim=1))                    # [N, 5, 1]

            pos_weights = (pos_weight_1, pos_weight_2)
            neg_weights = (neg_weight_1.squeeze(2), neg_weight_2.squeeze(2))
        else:
            neg_weights = None
            pos_weights = None

        return score, pos_weights, neg_weights

    def mil_score_v2(self, ent_feat, box_feat):
        """

        Args:
            ent_feat: [N, 16, 768]
            box_feat: [N, 36, 1536]

        Returns:
            score: [N, 16, N, 36]

        """
        ent_feat = self.text_encoder(ent_feat)[:, :, None, None]  # [N, 16, 1, 1, 768]
        box_feat = self.visual_encoder(box_feat)[None, None]      # [1, 1, N, 36, 768]
        ent_feat = normalize(ent_feat, dim=-1)                    # [N, 16, 1, 1, 768]
        box_feat = normalize(box_feat, dim=-1)                    # [1, 1, N, 36, 768]
        merged_feat = self.dropout(ent_feat * box_feat)           # [N, 16, N, 36, 768]
        score = self.classifier(merged_feat).squeeze(dim=-1)      # [N, 16, N, 36]

        return score
