#  Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

from torchvision.transforms import transforms

import clip

cudnn.benchmark = True

# Python imports
import tqdm
from tqdm import tqdm
import os
from os.path import join as ospj
import csv
from datetime import datetime
from timm import create_model

# Local imports
from data import dataset
from data import dataset_clip
from data import dataset_phosc_clip
from data import dataset_phosc_clip_new
from models.common import Evaluator
from utils.utils import save_args, load_args
from utils.config_model import configure_model
from flags import parser, DATA_FOLDER
from utils.dbe import dbe

from modules import models, residualmodels

best_auc = 0
best_hm = 0
compose_switch = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'



def main():
    # Get arguments and start logging
    args = parser.parse_args()
    load_args(args.config, args)

    logpath = os.path.join(args.cv_dir, args.name)
    os.makedirs(logpath, exist_ok=True)
    save_args(args, logpath, args.config)
    writer = SummaryWriter(log_dir=logpath, flush_secs=30)

    print(args.emb_init)

    # Set CompositDataset
    if args.emb_init == 'clip':
        # dset = dataset_clip.CompositionDataset
        dset = dataset_phosc_clip_new.CompositionDataset
    else:
        dset = dataset.CompositionDataset

    image_transform = transforms.ToTensor()

    # Define phosc model
    phosc_model = create_model(
        model_name=args.model_name,
        phos_size=args.phos_size,
        phoc_size=args.phoc_size,
        phos_layers=args.phos_layers,
        phoc_layers=args.phoc_layers,
        dropout=args.dropout
    ).to(device)
    
    phosc_model.load_state_dict(torch.load(args.pretrained_weights))

    clip_model, clip_transform = clip.load('ViT-B/32')

    # Get dataset
    trainset = dset(
        root=os.path.join(DATA_FOLDER, args.data_dir),
        phase='train',
        split=args.splitname,
        model=args.image_extractor,
        num_negs=args.num_negs,
        pair_dropout=args.pair_dropout,
        update_features=args.update_features,
        train_only=args.train_only,
        open_world=args.open_world,
        phosc_transorm=image_transform,
        phosc_model=phosc_model,
        clip_model=clip_model,
        clip_transform=clip_transform,
        args=args
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers
    )

    testset = dset(
        root=os.path.join(DATA_FOLDER, args.data_dir),
        phase=args.test_set,
        split=args.splitname,
        model=args.image_extractor,
        subset=args.subset,
        update_features=args.update_features,
        open_world=args.open_world,
        phosc_transorm=image_transform,
        phosc_model=phosc_model,
        clip_model=clip_model,
        clip_transform=clip_transform,
        args=args
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers
    )

    # Get model and optimizer
    image_extractor, model, optimizer = configure_model(args, trainset)
    args.extractor = image_extractor

    train = train_normal

    evaluator_val = Evaluator(testset, model)

    print(model)

    start_epoch = 0
    # Load checkpoint
    if args.load is not None:
        checkpoint = torch.load(args.load)
        if image_extractor:
            try:
                image_extractor.load_state_dict(checkpoint['image_extractor'])
                if args.freeze_features:
                    print('Freezing image extractor')
                    image_extractor.eval()
                    for param in image_extractor.parameters():
                        param.requires_grad = False
            except:
                print('No Image extractor in checkpoint')
        model.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        print('Loaded model from ', args.load)

    for epoch in tqdm(range(start_epoch, args.max_epochs + 1), desc='Current epoch'):
        train(epoch, image_extractor, model, trainloader, optimizer, writer)

        if model.is_open and args.model == 'compcos' and ((epoch + 1) % args.update_feasibility_every) == 0:
            print('Updating feasibility scores')
            model.update_feasibility(epoch + 1.)

        if epoch % args.eval_val_every == 0:
            with torch.no_grad():  # todo: might not be needed
                test(epoch, image_extractor, model, testloader, evaluator_val, writer, args, logpath)

    write_log(best_auc, best_hm)


def train_normal(epoch, image_extractor, model, trainloader, optimizer, writer):
    '''
    Runs training for an epoch
    '''

    if image_extractor:
        image_extractor.train()

    model.train()  # Let's switch to training

    train_loss = 0.0
    for idx, data in tqdm(enumerate(trainloader), total=len(trainloader), desc='Training'):
        # data = [d.to(device) for d in data]

        model_data = [
            data['image']['pred_image'].to(device),
            data['attr']['truth_idx'].to(device),
            data['obj']['truth_idx'].to(device),
            data['pairs']['all'].to(device)
        ]

        if image_extractor:
            data['image']['path'] = image_extractor(data['image']['path'])

        loss, _ = model(model_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss = train_loss / len(trainloader)
    writer.add_scalar('Loss/train_total', train_loss, epoch)
    print('Epoch: {}| Loss: {}'.format(epoch, round(train_loss, 2)))


def test(epoch, image_extractor, model, testloader, evaluator, writer, args, logpath):
    '''
    Runs testing for an epoch
    '''
    global best_auc, best_hm

    def save_checkpoint(filename):
        state = {
            'net': model.state_dict(),
            'epoch': epoch,
            'AUC': stats['AUC']
        }

        if image_extractor:
            state['image_extractor'] = image_extractor.state_dict()

        torch.save(state, os.path.join(logpath, 'ckpt_{}.t7'.format(filename)))

    if image_extractor:
        image_extractor.eval()

    model.eval()

    accuracies, all_sub_gt, all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], [], [], []

    for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing'):
        # FIX: At this point I get data where the ground truth is the converted values. I need to have them and the indexes of the values(I think)

        model_data = [
            data['image']['pred_image'].to(device),
            data['attr']['truth_idx'].to(device),
            data['obj']['truth_idx'].to(device),
            data['pairs']['all'].to(device)
        ]

        if image_extractor:
            data['image']['path'] = image_extractor(data['image']['path'])

        _, predictions = model(model_data)

        # dbe(predictions)

        attr_truth, obj_truth, pair_truth = (
            data['attr']['truth_idx'],
            data['obj']['truth_idx'],
            data['pairs']['truth_idx']
        )

        # FIX: Here is the problem (ish). attr_truth and obj_truth are the CLIP and PHOSC vectors not the actual words. This needs to be fixed
        # dbe(data, attr_truth, obj_truth, pair_truth)

        all_pred.append(predictions)
        all_attr_gt.append(attr_truth)
        all_obj_gt.append(obj_truth)
        all_pair_gt.append(pair_truth)

    if args.cpu_eval:
        all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt), torch.cat(all_obj_gt), torch.cat(all_pair_gt)
    else:
        all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt).to('cpu'), torch.cat(all_obj_gt).to(
            'cpu'
        ), torch.cat(all_pair_gt).to('cpu')

    all_pred_dict = {}
    # Gather values as dict of (attr, obj) as key and list of predictions as values
    if args.cpu_eval:
        for k in all_pred[0].keys():
            all_pred_dict[k] = torch.cat(
                [all_pred[i][k].to('cpu') for i in range(len(all_pred))]
            )
    else:
        for k in all_pred[0].keys():
            all_pred_dict[k] = torch.cat(
                [all_pred[i][k] for i in range(len(all_pred))]
            )


    # Calculate best unseen accuracy
    # dbe(all_obj_gt)
    # NOTE: Called here. all_obj_gt, needs to be the real value?
    results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=args.bias, topk=args.topk)
    stats = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict, topk=args.topk)

    stats['a_epoch'] = epoch

    result = ''
    # write to Tensorboard
    for key in stats:
        writer.add_scalar(key, stats[key], epoch)
        result = result + key + '  ' + str(round(stats[key], 4)) + '| '

    result = result + args.name
    print(f'Test Epoch: {epoch}')
    print(result)

    if epoch > 0 and epoch % args.save_every == 0:
        save_checkpoint(epoch)

    if stats['AUC'] > best_auc:
        best_auc = stats['AUC']
        print('New best AUC ', best_auc)
        save_checkpoint('best_auc')
        write_log(auc=best_auc)

    if stats['best_hm'] > best_hm:
        best_hm = stats['best_hm']
        print('New best HM ', best_hm)
        save_checkpoint('best_hm')
        write_log(hm=best_hm)

    # Logs
    with open(ospj(logpath, 'logs.csv'), 'a') as f:
        w = csv.DictWriter(f, stats.keys())
        if epoch == 0:
            w.writeheader()
        w.writerow(stats)


# Logging to a file in Python
def write_log(auc=None, hm=None):
    args = parser.parse_args()

    print('Best AUC achieved is ', auc)
    print('Best HM achieved is ', hm)

    log_file = 'logs/text/log_train_cgqa_open_world.txt'

    with open(log_file, 'a') as file:  # 'a' stands for 'append'
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # format datetime as string
        file.write(f'Data: {args.data_dir} | Time: {timestamp}\n')
        if auc:
            file.write(f'Best AUC achieved is: {str(auc)}\n')
        if hm:
            file.write(f'Best HM achieved is: {str(hm)}\n')
        file.write('\n')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        write_log(best_auc, best_hm)
