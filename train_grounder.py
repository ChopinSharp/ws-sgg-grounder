import json
import os
from time import time
from datetime import datetime
import copy

from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn.init import zeros_, xavier_uniform_
from torch.utils.tensorboard import SummaryWriter

from grounder import MILEntityGroundingDatasetV2
from grounder import AdaptiveSmoothLabelEntityGroundingDataset
from grounder import DropoutGrounderV3


cfg = dict(
    BATCH_SIZE_LABEL=2048,
    BATCH_SIZE_MIL=24,
    LR=1e-3,
    WD=1e-3,
    EPOCH=15,
    DROPOUT=0.2,
    MARGIN=0.2,
    SMOOTH_RATIO=0.1,
    DESC='smooth; soft label; aggregate by adaptive weights; dp 0.2; MIL loss on image level; mean of max; hinge loss, margin=0.2',
    TITLE='adaptive_smooth_mil_v2',
)

LOG_INTERVAL = 50
VAL_INTERVAL = 100

CUDA_ZERO_SCALAR = torch.tensor(0, dtype=torch.float).cuda()


def time_print(info):
    print('[{}] {}'.format(datetime.now().strftime('%Y-%m-%d-%H:%M:%S'), info))


def initialize_grounder(grounder):
    for name, param in grounder.named_parameters():
        if 'weight' in name:
            xavier_uniform_(param)
        elif 'bias' in name:
            zeros_(param)


def aggregate_label(label_1, label_2):
    return (label_1 + label_2) / 2


def repeat_loader(loader):
    while True:
        for data in loader:
            yield data


def compute_loss_and_accuracy_label(Flag, grounder, ent_feat, box_feat, label_1, label_2, valid_mask, training, hard_label, neg_label_1, neg_label_2):
    with torch.autograd.set_grad_enabled(training):
        label_1 = label_1.to('cuda:0')
        label_2 = label_2.to('cuda:0')
        ent_feat = ent_feat.to('cuda:0')
        box_feat = box_feat.to('cuda:0')
        neg_label_1 = neg_label_1.to('cuda:0')
        neg_label_2 = neg_label_2.to('cuda:0')
        valid_mask = valid_mask.to('cuda:0', dtype=torch.float)
        score, (pos_weight_1, pos_weight_2), (neg_weight_1, neg_weight_2) = grounder(ent_feat, box_feat, label_1, neg_label_1, label_2, neg_label_2)
        score = valid_mask * score + (1 - valid_mask) * (-1e7)
        log_prob = torch.nn.functional.log_softmax(score, dim=1)

        """adaptive1"""
        loss_1 = torch.max(torch.sigmoid(neg_weight_2) - torch.sigmoid(pos_weight_1)[:, None] + 0.2, torch.tensor(0.0).cuda()).mean()
        loss_2 = torch.max(torch.sigmoid(neg_weight_1) - torch.sigmoid(pos_weight_2)[:, None] + 0.2, torch.tensor(0.0).cuda()).mean()

    with torch.no_grad():
        '''softmax'''
        weights = nn.functional.softmax(torch.cat([pos_weight_1, pos_weight_2], dim=1), dim=1)       # [N, 2]

        if Flag:
            merged_label = label_1 * weights[:, :1] + label_2 * weights[:, 1:]                   # [N, 36]
        else:
            merged_label = aggregate_label(label_1, label_2)

    with torch.autograd.set_grad_enabled(training):
        loss_t = nn.functional.kl_div(log_prob, merged_label, reduction='none')
        loss = torch.mean(torch.sum(loss_t * valid_mask, dim=1), dim=0)
        loss = loss + loss_1 + loss_2

    with torch.no_grad():
        sorted_idx = torch.argsort(score, dim=1, descending=True)
        sample_idx = torch.arange(len(score)).unsqueeze(dim=1).repeat_interleave(5, dim=1)
        recall_dict = {}
        for K in [1, 3, 5]:
            pred_pos = torch.zeros_like(score)
            pred_pos[sample_idx[:, :K].flatten(), sorted_idx[:, :K].flatten()] = 1.
            true_pos = (hard_label > 0).to('cuda:0')
            recall_t = torch.sum(pred_pos * true_pos, dim=1) / torch.sum(true_pos, dim=1)
            recall_dict[K] = recall_t.mean().item()

    return loss, recall_dict


def compute_loss_mil(grounder, ent_feat, box_feat, valid_mask, hard_label, training):

    N = ent_feat.shape[0]
    with torch.no_grad():
        pos_mask = torch.eye(N, device='cuda:0')
        neg_mask = 1 - pos_mask
        pos_idx = pos_mask > 0

    with torch.autograd.set_grad_enabled(training):
        ent_feat = ent_feat.to('cuda:0')                       # [N, ent_num, 768 ]
        box_feat = box_feat.to('cuda:0')                       # [N, 36, 1536]
        score = grounder.mil_score_v2(ent_feat, box_feat)      # [N, ent_num N, 36]
        mean_max_score = score.max(dim=-1)[0].mean(dim=1)      # [N, N]

        # loss = torch.nn.functional.cross_entropy(mean_max_score, torch.arange(N).cuda())

        pos_score = mean_max_score[pos_idx]    # [N, 1]
        neg_score = mean_max_score * neg_mask  # [N, N]
        loss = torch.max(neg_score-pos_score[:, None] + cfg['MARGIN'], CUDA_ZERO_SCALAR).mean()

    return loss


def mil_decay_factor(epoch):
    if epoch < 5:
        return 1.
    elif epoch < 10:
        return 1. - (epoch - 4) * 0.2
    else:
        return 0.


def train():
    ckpt = None
    if 'RESUME' in cfg:
        ckpt_path = os.path.join(cfg['RESUME']['DIR'], 'grounder_ckpt_epoch_{}.pth'.format(cfg['RESUME']['EPOCH']))
        ckpt = torch.load(ckpt_path)
        print('Loaded checkpoint from {}.'.format(ckpt_path))
    timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    experiment_id = '{}-{}'.format(timestamp, cfg['TITLE'])
    output_dir = 'output/{}'.format(experiment_id)
    os.makedirs(output_dir)
    # Build dataloaders
    label_trn_dataset = AdaptiveSmoothLabelEntityGroundingDataset(split='train')
    label_val_dataset = AdaptiveSmoothLabelEntityGroundingDataset(split='val')
    label_trn_loader = DataLoader(label_trn_dataset, batch_size=cfg['BATCH_SIZE_LABEL'], shuffle=True, num_workers=16)
    label_val_loader = DataLoader(label_val_dataset, batch_size=cfg['BATCH_SIZE_LABEL'], shuffle=True, num_workers=16)
    mil_trn_dataset = MILEntityGroundingDatasetV2(split='train', smooth_ratio=cfg['SMOOTH_RATIO'])
    # mil_val_dataset = MILEntityGroundingDatasetV2(split='val', smooth_ratio=cfg['SMOOTH_RATIO'])
    mil_trn_loader = DataLoader(mil_trn_dataset, batch_size=cfg['BATCH_SIZE_MIL'], shuffle=True, num_workers=16)
    # mil_val_loader = DataLoader(mil_val_dataset, batch_size=cfg['BATCH_SIZE'], shuffle=True, num_workers=16)
    mil_trn_loader = repeat_loader(mil_trn_loader)
    # Tensorboard writer
    tb_dir = 'tb/{}'.format(experiment_id)
    label_trn_writer = SummaryWriter(os.path.join(tb_dir, 'label', 'train'))
    label_val_writer = SummaryWriter(os.path.join(tb_dir, 'label', 'val'))
    mil_trn_writer = SummaryWriter(os.path.join(tb_dir, 'mil', 'train'))
    # mil_val_writer = SummaryWriter(os.path.join(tb_dir, 'mil', 'val'))
    # Build and init grounder
    grounder = DropoutGrounderV3(dropout_p=cfg['DROPOUT'])
    if ckpt is not None:
        grounder.load_state_dict(ckpt['model'])
    else:
        initialize_grounder(grounder)
    grounder.to('cuda:0')
    # Setup optimizer
    optimizer = optim.Adam(grounder.parameters(), lr=cfg['LR'], weight_decay=cfg['WD'])
    if ckpt is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, min_lr=cfg['LR']/500, mode='min', factor=0.5,
        verbose=True, threshold_mode='rel', patience=1, threshold=5e-3)
    if ckpt is not None:
        scheduler.load_state_dict(ckpt['scheduler'])
    # Start training
    step = 0
    label_trn_running_loss = mil_trn_running_loss = trn_running_recall_1 = trn_running_recall_3 = trn_running_recall_5 = 0.
    best_model = {'avg_val_loss': float('inf')}
    tic = time()
    start_from = 0 if 'RESUME' not in cfg else cfg['RESUME']['EPOCH']
    for epoch in range(start_from, start_from + cfg['EPOCH']):
        for label_trn_batch in label_trn_loader:
            # Train for one step
            step += 1
            grounder.train()
            optimizer.zero_grad()
            mil_trn_batch = next(mil_trn_loader)
            mil_trn_loss = compute_loss_mil(
                grounder=grounder, training=True, **mil_trn_batch
            )
            decayed_mil_trn_loss = mil_trn_loss * mil_decay_factor(epoch)
            decayed_mil_trn_loss.backward()
            mil_trn_running_loss += mil_trn_loss.item()
            label_trn_loss, trn_recall = compute_loss_and_accuracy_label(
                epoch > 5, grounder=grounder, training=True, **label_trn_batch
            )
            label_trn_loss.backward()
            label_trn_running_loss += label_trn_loss.item()
            trn_running_recall_1 += trn_recall[1]
            trn_running_recall_3 += trn_recall[3]
            trn_running_recall_5 += trn_recall[5]
            optimizer.step()
            # Log training loss
            if step % LOG_INTERVAL == 0:
                mil_avg_trn_loss = mil_trn_running_loss / LOG_INTERVAL
                label_avg_trn_loss = label_trn_running_loss / LOG_INTERVAL
                avg_trn_recall_1 = trn_running_recall_1 / LOG_INTERVAL
                avg_trn_recall_3 = trn_running_recall_3 / LOG_INTERVAL
                avg_trn_recall_5 = trn_running_recall_5 / LOG_INTERVAL
                time_print('epoch {} step {}: mil_loss={:.6f}, label_loss={:.6f}, R @ 1={:.4f}, R @ 3={:.4f}, R @ 5={:.4f}'.format(
                    epoch + 1, step, mil_avg_trn_loss, label_avg_trn_loss, avg_trn_recall_1, avg_trn_recall_3, avg_trn_recall_5))
                mil_trn_writer.add_scalar('loss', mil_avg_trn_loss, step)
                label_trn_writer.add_scalar('loss', label_avg_trn_loss, step)
                label_trn_writer.add_scalar('R @ 1', avg_trn_recall_1, step)
                label_trn_writer.add_scalar('R @ 3', avg_trn_recall_3, step)
                label_trn_writer.add_scalar('R @ 5', avg_trn_recall_5, step)
                mil_trn_running_loss = label_trn_running_loss = trn_running_recall_1 = trn_running_recall_3 = trn_running_recall_5 = 0.
            # Eval on whole val split
            if step % VAL_INTERVAL == 0:
                # Compute and log val loss
                grounder.eval()
                val_loss_list, val_recall_1_list, val_recall_3_list, val_recall_5_list = [], [], [], []
                pbar = tqdm(total=len(label_val_dataset), ascii=True, desc='validating', ncols=120)
                for val_batch in label_val_loader:
                    loss, val_recall = compute_loss_and_accuracy_label(
                        epoch > 5, grounder=grounder, training=False, **val_batch
                    )
                    val_loss_list.append(loss.item())
                    val_recall_1_list.append(val_recall[1])
                    val_recall_3_list.append(val_recall[3])
                    val_recall_5_list.append(val_recall[5])
                    pbar.update(val_batch['valid_mask'].size(0))
                pbar.close()
                avg_val_loss = sum(val_loss_list) / len(val_loss_list)
                avg_val_recall_1 = sum(val_recall_1_list) / len(val_recall_1_list)
                avg_val_recall_3 = sum(val_recall_3_list) / len(val_recall_3_list)
                avg_val_recall_5 = sum(val_recall_5_list) / len(val_recall_5_list)
                time_print('*VAL* epoch {} step {}: label_loss={:.6f}, R @ 1={:.4f}, R @ 3={:.4f}, R @ 5={:.4f}\n'.format(
                    epoch + 1, step, avg_val_loss, avg_val_recall_1, avg_val_recall_3, avg_val_recall_5))
                label_val_writer.add_scalar('loss', avg_val_loss, step)
                label_val_writer.add_scalar('R @ 1', avg_val_recall_1, step)
                label_val_writer.add_scalar('R @ 3', avg_val_recall_3, step)
                label_val_writer.add_scalar('R @ 5', avg_val_recall_5, step)
                # Update learning rate
                scheduler.step(avg_val_loss)
                # Track model with lowest val loss
                if avg_val_loss < best_model['avg_val_loss']:
                    best_model['avg_val_loss'] = avg_val_loss
                    best_model['avg_val_recall_1'] = avg_val_recall_1
                    best_model['avg_val_recall_3'] = avg_val_recall_3
                    best_model['avg_val_recall_5'] = avg_val_recall_5
                    best_model['epoch'] = epoch + 1
                    best_model['step'] = step
                    best_model['weights'] = copy.deepcopy(grounder.state_dict())
        # Save checkpoint after each epoch
        epoch_ckpt = {
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'model': grounder.state_dict()
        }
        save_path = os.path.join(output_dir, 'grounder_ckpt_epoch_{}.pth'.format(epoch + 1))
        torch.save(epoch_ckpt, save_path)
    # Save best model
    time_spent = int(time() - tic) // 60
    time_print('Training completed in {} h {} m.'.format(time_spent // 60, time_spent % 60))
    time_print('Found model with lowest val loss at epoch {epoch} step {step}.'.format(**best_model))
    save_path = os.path.join(output_dir, 'grounder_best.pth')
    torch.save(best_model['weights'], save_path)
    time_print('Saved best model weights to {}'.format(save_path))
    # Close summary writer
    label_trn_writer.close()
    label_val_writer.close()
    mil_trn_writer.close()
    # Log training procedure
    model_info = {
        'type': 'DropoutGrounderV3',
        'config': cfg
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(model_info, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    train()
