#  Torch imports
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import numpy as np
from flags import DATA_FOLDER

from torchvision.transforms import transforms

cudnn.benchmark = True

# Python imports
import tqdm
from tqdm import tqdm
import os
from os.path import join as ospj
from timm import create_model
import clip

# Local imports
from data import dataset as dset
from models.common import Evaluator
from utils.utils import load_args
from utils.config_model import configure_model
from flags import parser

from data import dataset
from data import dataset_phosc_clip_new

from modules import models, residualmodels

from utils.dbe import dbe

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    # Get arguments and start logging
    args = parser.parse_args()
    logpath = args.logpath
    config = [os.path.join(logpath, _) for _ in os.listdir(logpath) if _.endswith('yml')][0]
    load_args(config, args)

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
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase='train',
        split=args.splitname,
        model=args.image_extractor,
        update_features=args.update_features,
        train_only=args.train_only,
        subset=args.subset,
        open_world=args.open_world,
        phosc_model=phosc_model,
        clip_model=clip_model,
        clip_transform=clip_transform,
        args=args
    )

    valset = dset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase='val',
        split=args.splitname,
        model=args.image_extractor,
        subset=args.subset,
        update_features=args.update_features,
        open_world=args.open_world,
        phosc_model=phosc_model,
        clip_model=clip_model,
        clip_transform=clip_transform,
        args=args
    )

    valoader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=8)

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

    args.model = 'compcos'

    # dbe(args)

    # Get model and optimizer
    image_extractor, model, optimizer = configure_model(args, trainset)
    args.extractor = image_extractor

    # args.load = ospj(logpath,'ckpt_best_auc.t7')
    args.load = ospj(logpath,'ckpt_best_hm.t7')

    checkpoint = torch.load(args.load)
    if image_extractor:
        try:
            image_extractor.load_state_dict(checkpoint['image_extractor'])
            image_extractor.eval()
        except:
            print('No Image extractor in checkpoint')
            
    # dbe(checkpoint['net'], checkpoint['net'].keys())
    model.load_state_dict(checkpoint['net'])
    model.eval()

    threshold = None
    if args.open_world and args.hard_masking:
        assert args.model == 'compcos', args.model + ' does not have hard masking.'

        if args.threshold is not None:
            threshold = args.threshold
        else:
            evaluator_val = Evaluator(valset, model)
            unseen_scores = model.compute_feasibility().to('cpu')
            seen_mask = model.seen_mask.to('cpu')

            min_feasibility = (unseen_scores+seen_mask*10.).min()
            max_feasibility = (unseen_scores-seen_mask*10.).max()

            thresholds = np.linspace(min_feasibility,max_feasibility, num=args.threshold_trials)

            best_auc = 0.
            best_th = -10

            with torch.no_grad():
                for th in thresholds:
                    results = test(image_extractor,model,valoader,evaluator_val,args,threshold=th,print_results=False)
                    auc = results['AUC']

                    if auc > best_auc:
                        best_auc = auc
                        best_th = th

                        print('New best AUC',best_auc)
                        print('Threshold',best_th)

            threshold = best_th

    evaluator = Evaluator(testset, model)

    # dbe(evaluator.test_pair_dict)
    with torch.no_grad():
        test(image_extractor, model, testloader, evaluator, args, threshold)


def test(image_extractor, model, testloader, evaluator,  args, threshold=None, print_results=True):
        if image_extractor:
            image_extractor.eval()

        model.eval()

        accuracies, all_sub_gt, all_attr_gt_idx, all_obj_gt_idx, all_pair_gt_idx, all_pred = [], [], [], [], [], []
        real_attr_gt, real_obj_gt = [], []

        for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing'):
            model_data = [
                data['image']['pred_image'].to(device),
                data['attr']['truth_idx'].to(device),
                data['obj']['truth_idx'].to(device),
                data['pairs']['all'].to(device),
                # TODO: Add acutal data here aswel, so this can be used in the `forward()` functions.
                data['attr']['pred'],
                data['obj']['pred']
            ]

            if threshold is None:
                _, predictions = model(model_data)
            else:
                _, predictions = model.val_forward_with_threshold(model_data, threshold)

            attr_truth_idx, obj_truth_idx, pair_truth_idx = (
                data['attr']['truth_idx'],
                data['obj']['truth_idx'],
                data['pairs']['truth_idx']
            )

            length = predictions[next(iter(predictions))].shape[1]

            # NOTE: Temp fix for dimention problems
            # FIX: Might want to use CLIP on the value where the index is pointing, then the dimentions will be correct. In `common` after the flatten compare the size of the CLIP. So the lenght of the CLIP is the lenght for eacn single element.
            attr_truth_idx = attr_truth_idx[0].unsqueeze(0).expand(length, -1).clone().squeeze()

            # NOTE: Temp fix for dimention problems
            obj_truth_idx = obj_truth_idx[0].unsqueeze(0).expand(length, -1).clone().squeeze()

            # NOTE: Temp fix for dimention problems
            pair_truth_idx = pair_truth_idx[0].unsqueeze(0).expand(length, -1).clone().squeeze()

            all_pred.append(predictions)
            all_attr_gt_idx.append(attr_truth_idx)
            all_obj_gt_idx.append(obj_truth_idx)
            all_pair_gt_idx.append(pair_truth_idx)

        if args.cpu_eval:
            all_attr_gt_idx, all_obj_gt_idx, all_pair_gt_idx = torch.cat(all_attr_gt_idx), torch.cat(all_obj_gt_idx), torch.cat(all_pair_gt_idx)
        else:
            all_attr_gt_idx = torch.cat(all_attr_gt_idx).to(device)  # torch.Size([420])

            all_obj_gt_idx = torch.cat(all_obj_gt_idx).to(device)    # torch.Size([420])
            all_pair_gt_idx = torch.cat(all_pair_gt_idx).to(device)  # torch.Size([420])

        all_pred_dict = {}
        # Gather values as dict of (attr, obj) as key and list of predictions as values
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

        # Calculate best unseen accuracy
        print('Score model:')
        results = evaluator.score_model(all_pred_dict, all_obj_gt_idx, bias=args.bias, topk=args.topk)

        print('Evaluate predictions:')
        stats = evaluator.evaluate_predictions(results, all_attr_gt_idx, all_obj_gt_idx, all_pair_gt_idx, all_pred_dict, topk=args.topk)

        result = ''
        for key in stats:
            result = result + key + '  ' + str(round(stats[key], 4)) + '| '

        result = result + args.name
        if print_results:
            print(f'Results')
            print(result)

        return results


if __name__ == '__main__':
    main()