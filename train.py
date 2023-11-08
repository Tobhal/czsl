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
        p=False,
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


def train_normal(epoch, image_extractor, model, train_loader, optimizer, writer):
    '''
    Runs training for an epoch
    '''

    if image_extractor:
        image_extractor.train()

    model.train()  # Let's switch to training

    train_loss = 0.0
    for idx, data in tqdm(enumerate(train_loader), total=len(train_loader), desc='Training'):
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

    train_loss = train_loss / len(train_loader)
    writer.add_scalar('Loss/train_total', train_loss, epoch)
    print('Epoch: {}| Loss: {}'.format(epoch, round(train_loss, 2)))


def test(epoch, image_extractor, model, test_loader, evaluator, writer, args, logpath):
    '''
    Runs testing for an epoch
    '''
    global best_auc, best_hm

    def save_checkpoint(filename):

        dbe(model.state_dict().keys(), should_exit=False)

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

    accuracies, all_sub_gt, all_attr_gt_idx, all_obj_gt_idx, all_pair_gt_idx, all_pred = [], [], [], [], [], []
    real_attr_gt, real_obj_gt = [], []

    for idx, data in tqdm(enumerate(test_loader), total=len(test_loader), desc='Testing'):
        # FIX: At this point I get data where the ground truth is the converted values. I need to have them and the indexes of the values(I think)

        # dbe(data['obj']['truth'], data['obj']['pred'], data['obj']['clip'])

        model_data = [
            data['image']['pred_image'].to(device),
            data['attr']['truth_idx'].to(device),
            data['obj']['truth_idx'].to(device),
            data['pairs']['all'].to(device),
            # TODO: Add acutal data here aswel, so this can be used in the `forward()` functions.
            data['attr']['pred'],
            data['obj']['pred']
        ]

        if image_extractor:
            data['image']['path'] = image_extractor(data['image']['path'])

        _, predictions = model(model_data)  # Len = 250

        # NOTE: Squeeze the extra dimention away so this prediction shape equals the prediction shape from the original czsl code
        predictions = {key: value.squeeze() for key, value in predictions.items()}

        attr_truth_idx, obj_truth_idx, pair_truth_idx = (
            data['attr']['truth_idx'],
            data['obj']['truth_idx'],
            data['pairs']['truth_idx']
        )

        # """
        length = predictions[next(iter(predictions))].shape[0]

        # dbe(attr_truth_idx, obj_truth_idx, pair_truth_idx)

        # NOTE: Temp fix for dimention problems
        # FIX: Might want to use CLIP on the value where the index is pointing, then the dimentions will be correct. In `common` after the flatten compare the size of the CLIP. So the lenght of the CLIP is the lenght for eacn single element.
        attr_truth_idx = attr_truth_idx[0].unsqueeze(0).expand(length, -1).clone().squeeze()

        # NOTE: Temp fix for dimention problems
        obj_truth_idx = obj_truth_idx[0].unsqueeze(0).expand(length, -1).clone().squeeze()

        # NOTE: Temp fix for dimention problems
        pair_truth_idx = pair_truth_idx[0].unsqueeze(0).expand(length, -1).clone().squeeze()
        # """

        all_pred.append(predictions)
        all_attr_gt_idx.append(attr_truth_idx)
        all_obj_gt_idx.append(obj_truth_idx)
        all_pair_gt_idx.append(pair_truth_idx)

        real_attr_gt.append(data['attr']['clip'])
        real_obj_gt.append(data['obj']['clip'])


    # TODO: Iterate each element of `all_attr_gt_idx`, `all_obj_gt_idx` and `all_pair_gt_idx` and copy the value from each tensor to a size of `len(predictions)` = `250`?

    if args.cpu_eval:
        all_attr_gt_idx, all_obj_gt_idx, all_pair_gt_idx = torch.cat(all_attr_gt_idx), torch.cat(all_obj_gt_idx), torch.cat(all_pair_gt_idx)
    else:
        all_attr_gt_idx = torch.cat(all_attr_gt_idx).to(device)  # torch.Size([420])

        all_obj_gt_idx = torch.cat(all_obj_gt_idx).to(device)    # torch.Size([420])
        all_pair_gt_idx = torch.cat(all_pair_gt_idx).to(device)  # torch.Size([420])

        real_attr_gt = torch.cat(real_attr_gt, dim=0).to(device) # torch.Size([420, 1, 77])
        real_obj_gt = torch.cat(real_obj_gt, dim=0).to(device)   # torch.Size([420, 13, 77])

        # dbe(len(all_pred))

    # dbe(len(all_pred[0].keys()), len(all_pred), len(all_pred[0].keys()) + len(all_pred))

    # dbe(all_obj_gt_idx, all_obj_gt_idx.shape)

    all_pred_dict = {}
    # Gather values as dict of (attr, obj) as key and list of predictions as values
    # NOTE: This is where the convertion to ([420, 1, 250]) happens
    # TODO: Find out how to ether get the `real_obj_gt` and `real_attr_gt` to be ([420, 1, 250]) or `all_pred_dict` to be ([420, 1, 77])
    if args.cpu_eval:
        for k in all_pred[0].keys():
            all_pred_dict[k] = torch.cat(
                [all_pred[i][k].to('cpu') for i in range(len(all_pred))]
            )
    else:
        for k in all_pred[0].keys():            
            temp_list = []
            for i in range(len(all_pred)):  # 420
                temp_list.append(all_pred[i][k])

            all_pred_dict[k] = torch.cat(temp_list)  # torch.Shape([420, 250])

    # `all_pred` = 420
    # `all_pred_dict` = 250

    # Calculate best unseen accuracy
    # dbe(all_obj_gt)
    # NOTE: Called here. all_obj_gt, needs to be the real value?

    print('Score model:')
    results = evaluator.score_model(all_pred_dict, all_obj_gt_idx, bias=args.bias, topk=args.topk)
    """
    {
        'open': (
            torch.Size([105000, 1]) | torch.Size([420, 1, 250])
            ,
            torch.Size([105000, 1]) | torch.Size([420, 1, 250])
        ),
        'unbiased_open': (
            torch.Size([105000, 1]) | torch.Size([420, 1, 250])
            ,
            torch.Size([105000, 1]) | torch.Size([420, 1, 250])
        ),
        'closed': (
            torch.Size([105000, 1]) | torch.Size([420, 1, 250])
            ,
            torch.Size([105000, 1]) | torch.Size([420, 1, 250])
        ),
        'unbiased_closed': (
            torch.Size([105000, 1]) | torch.Size([420, 1, 250])
            ,
            torch.Size([105000, 1]) | torch.Size([420, 1, 250])
        ),
        'object_oracle': (
            torch.Size([105000, 1]) | torch.Size([420, 1, 250])
            ,
            torch.Size([105000, 1]) | torch.Size([420, 1, 250])
        ),
        'object_oracle_unbiased': (
            torch.Size([105000, 1]) | torch.Size([420, 1, 250])
            ,
            torch.Size([105000, 1]) | torch.Size([420, 1, 250])
        ),
        'scores': torch.Size([420, 250, 250])
    }
    """

    # dbe(all_attr_gt_idx, all_attr_gt_idx.shape, results['object_oracle'][0][:, 0], results['object_oracle'][0][:, 0].shape)
    # stats = evaluator.evaluate_predictions(results, real_attr_gt, all_obj_gt_idx, all_pair_gt_idx, all_pred_dict, topk=args.topk)
    print('Evaluate predictions:')
    stats = evaluator.evaluate_predictions(results, all_attr_gt_idx, all_obj_gt_idx, all_pair_gt_idx, all_pred_dict, topk=args.topk)

    stats['a_epoch'] = epoch

    result = ''
    # write to Tensorboard
    for key in stats:
        writer.add_scalar(key, stats[key], epoch)
        result = result + key + '  ' + str(round(stats[key], 4)) + '| '

    result = result + args.name
    print(f'Test Epoch: {epoch}')
    print(result)

    if epoch >= 0 and epoch % args.save_every == 0:
        save_checkpoint(epoch)

    if epoch >= 0 or stats['AUC'] > best_auc:
        best_auc = stats['AUC']
        print('New best AUC ', best_auc)
        save_checkpoint('best_auc')
        write_log(auc=best_auc)

    if epoch >= 0 or stats['best_hm'] > best_hm:
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
