#external libs
import numpy as np
from tqdm import tqdm
from PIL import Image
import clip
import os
import random
from os.path import join as ospj
from glob import glob 
#torch libs
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
#local libs
from utils.utils import get_norm_values, chunks
from models.image_extractor import get_image_extractor
from itertools import product

# Phosc
from modules.utils import generate_phos_vector, generate_phoc_vector, set_phos_version, set_phoc_version, gen_shape_description

from utils.dbe import dbe, file_exists

from timm import create_model

# Typehinting
from typing import Union
from os import PathLike

import random

import torch.multiprocessing as mp
mp.set_start_method('spawn')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ImageLoader:
    """
    Loades a image from a given path and converts it to RPG
    """
    def __init__(self, root: PathLike):
        self.root_dir = root

    def __call__(self, img: str):
        img = Image.open(ospj(self.root_dir,img)).convert('RGB') #We don't want alpha
        return img


def dataset_transform(phase, norm_family = 'imagenet'):
    '''
        Transforms a image pased on the phase

        Inputs
            phase: String controlling which set of transforms to use
            norm_family: String controlling which normaliztion values to use
        
        Returns
            transform: A list of pytorch transforms
    '''
    mean, std = get_norm_values(norm_family=norm_family)

    if phase == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    elif phase == 'val' or phase == 'test':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif phase == 'all':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise ValueError('Invalid transform')

    return transform

def filter_data(all_data, pairs_gt, topk = 5):
    '''
    Helper function to clean data
    '''
    valid_files = []
    with open('/home/ubuntu/workspace/top'+str(topk)+'.txt') as f:
        for line in f:
            valid_files.append(line.strip())

    data, pairs, attr, obj  = [], [], [], []
    for current in all_data:
        if current[0] in valid_files:
            data.append(current)
            pairs.append((current[1],current[2]))
            attr.append(current[1])
            obj.append(current[2])
            
    counter = 0
    for current in pairs_gt:
        if current in pairs:
            counter+=1
    print('Matches ', counter, ' out of ', len(pairs_gt))
    print('Samples ', len(data), ' out of ', len(all_data))
    return data, sorted(list(set(pairs))), sorted(list(set(attr))), sorted(list(set(obj)))

# Dataset class now
class CompositionDataset(Dataset):
    '''
    Inputs
        root: String of base dir of dataset
        phase: String train, val, test
        split: String dataset split
        subset: Boolean if true uses a subset of train at each epoch
        num_negs: Int, numbers of negative pairs per batch
        pair_dropout: Percentage of pairs to leave in current epoch
    '''
    def __init__(
        self,
        root: PathLike,
        phase: str,
        phosc_model,
        clip_model,
        clip_transform,
        split = 'compositional-split',
        model = 'resnet18',
        norm_family = 'imagenet',
        subset = False,
        num_negs = 1,
        pair_dropout = 0.0,
        update_features = False,
        return_images = False,
        train_only = False,
        open_world=False,
        phosc_transorm=None,
        p=False,
        **args
    ):
        self.root = root
        self.phase = phase
        self.split = split
        self.num_negs = num_negs
        self.pair_dropout = pair_dropout
        self.norm_family = norm_family
        self.return_images = return_images
        self.update_features = update_features
        self.feat_dim = 512 if 'resnet18' in model else 2048 # todo, unify this  with models
        self.open_world = open_world
        self.phosc_transorm = phosc_transorm

        self.args = args['args']

        self.phosc_model = phosc_model
        self.clip_model = clip_model
        self.clip_transform = clip_transform

        # Make a clip representation of the language word that is used in the dataset
        
        self.clip_language_text = clip.tokenize(self.args.language_name)

        # Sett phos and phoc language
        set_phos_version(self.args.phosc_version)
        set_phoc_version(self.args.phosc_version)        

        self.attrs, self.objs, self.pairs, self.train_pairs, self.val_pairs, self.test_pairs, self.train_data, self.val_data, self.test_data = self.parse_split()
        
        if p:
            dbe(len(self.attrs), len(self.objs))

        self.full_pairs = list(product(self.attrs, self.objs))
        
        # NOTE: Change the obj2idx and attr2idx to actualy be the data we want.
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}

        if self.open_world:
            self.pairs = self.full_pairs

        self.all_pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        # NOTE: Change the pair2idx to be the actual data we want.
        if train_only and self.phase == 'train':
            print('Using only train pairs')
            self.pair2idx = {pair : idx for idx, pair in enumerate(self.train_pairs)}
        else:
            print('Using all pairs')
            self.pair2idx = {pair : idx for idx, pair in enumerate(self.pairs)}

        # Setting data to phase
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        elif self.phase == 'test':
            self.data = self.test_data
        elif self.phase == 'all':
            print('Using all data')
            self.data = self.train_data + self.val_data + self.test_data
        else:
            raise ValueError('Invalid training phase')

        self.all_data = self.train_data + self.val_data + self.test_data

        print('Dataset loaded')
        print('Train pairs: {}, Validation pairs: {}, Test Pairs: {}'.format(
            len(self.train_pairs), len(self.val_pairs), len(self.test_pairs))
        )
        print('Train images: {}, Validation images: {}, Test images: {}'.format(
            len(self.train_data), len(self.val_data), len(self.test_data))
        )
        
        if subset:
            ind = np.arange(len(self.data))
            ind = ind[::len(ind) // 1000]
            self.data = [self.data[i] for i in ind]

        # Keeping a list of all pairs that occur with each object
        self.obj_affordance = {}
        self.train_obj_affordance = {}
        for _obj in self.objs:
            candidates = [attr for (_, attr, obj) in self.train_data+self.test_data if obj==_obj]
            self.obj_affordance[_obj] = list(set(candidates))

            candidates = [attr for (_, attr, obj) in self.train_data if obj==_obj]
            self.train_obj_affordance[_obj] = list(set(candidates))

        self.sample_indices = list(range(len(self.data)))

        self.sample_pairs = self.train_pairs

        # Load based on what to output
        self.transform = dataset_transform(self.phase, self.norm_family)

        self.loader = ImageLoader(ospj(self.root, self.args.splitname))

        if not self.update_features:
            feat_file = ospj(root, f'{model}_{self.args.splitname}_featurers.t7')
            print(f'Using {model} and feature file {feat_file}')

            if not os.path.exists(feat_file):
                with torch.no_grad():
                    self.generate_features(feat_file, model)

            self.phase = phase
            activation_data = torch.load(feat_file)

            self.activations = dict(
                zip(activation_data['files'], activation_data['features'])
            )

            self.feat_dim = activation_data['features'].size(1)

            print('{} activations loaded'.format(len(self.activations)))



    def parse_split(self):
        """
        TODO: Rework

        This should return the State (the word 'bengali') and the bengali word. The phos calculations will happen later.
        Do this for the train, test and validation data csv files. Do not sort the list so the data can be found later using the index.

        example of a pair: ['Bengali', 'চান্দুরি']

        NOTE: This is sorted, this can liead to problems?
        """

        '''
        Helper function to read splits of object atrribute pair
        Returns
            all_attrs: List of all attributes
            all_objs: List of all objects
            all_pairs: List of all combination of attrs and objs
            tr_pairs: List of train pairs of attrs and objs
            vl_pairs: List of validation pairs of attrs and objs
            ts_pairs: List of test pairs of attrs and objs
        '''
        def parse_pairs(pair_list: Union[str, PathLike], phase: str):
            '''
            Helper function to parse each phase to object attrribute vectors

            Inputs
                pair_list: path to textfile
            '''
            with open(ospj(pair_list, phase, 'data.csv'), 'r') as f:
                lines = f.read().strip().split('\n')
                pairs = lines[1:]   # Skip first line (headers)
                pairs = [line.split(',') for line in pairs]

                data = []

                for image, word in tqdm(pairs, desc=f'Prepearing pairs, {phase}'):
                    data.append([
                        # Image path
                        ospj(phase, image),
                        # Attrs
                        self.args.language_name,
                        # Obj
                        word
                    ])

                pairs = [[self.args.language_name, line[1]] for line in pairs]
                pairs = list(map(tuple, pairs))

            attrs, objs = zip(*pairs)

            return attrs, objs, pairs, data

        # Train
        tr_attrs, tr_objs, tr_pairs, tr_data = parse_pairs(
            ospj(self.root, self.split),
            'train'
        )

        # Validation
        vl_attrs, vl_objs, vl_pairs, vl_data = parse_pairs(
            ospj(self.root, self.split),
            'val'
        )

        # Test
        ts_attrs, ts_objs, ts_pairs, ts_data = parse_pairs(
            ospj(self.root, self.split),
            'test'
        )
        
        #now we compose all objs, attrs and pairs
        all_attrs, all_objs = sorted(
            list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
                list(set(tr_objs + vl_objs + ts_objs))
            )
        
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs, tr_data, vl_data, ts_data


    def get_dict_data(self, data, pairs) -> dict:
        data_dict = {}
        for current in pairs:
            data_dict[current] = []

        for current in data:
            image, attr, obj = current
            data_dict[(attr, obj)].append(image)
        
        return data_dict


    def reset_dropout(self):
        ''' 
        Helper function to sample new subset of data containing a subset of pairs of objs and attrs
        '''
        self.sample_indices = list(range(len(self.data)))
        self.sample_pairs = self.train_pairs

        # Using sampling from random instead of 2 step numpy
        n_pairs = int((1 - self.pair_dropout) * len(self.train_pairs))

        self.sample_pairs = random.sample(self.train_pairs, n_pairs)
        print('Sampled new subset')
        print('Using {} pairs out of {} pairs right now'.format(
            n_pairs, len(self.train_pairs)))

        self.sample_indices = [ i for i in range(len(self.data))
            if (self.data[i][1], self.data[i][2]) in self.sample_pairs
        ]
        print('Using {} images out of {} images right now'.format(
            len(self.sample_indices), len(self.data)))

    def sample_negative(self, attr, obj) -> tuple:
        '''
        Inputs
            attr: String of valid attribute
            obj: String of valid object
        Returns
            Tuple of a different attribute, object indexes
        '''
        new_attr, new_obj = self.sample_pairs[np.random.choice(
            len(self.sample_pairs))]

        while new_attr == attr and new_obj == obj:
            new_attr, new_obj = self.sample_pairs[np.random.choice(
            len(self.sample_pairs))]
        
        return (self.attr2idx[new_attr], self.obj2idx[new_obj])

    def sample_affordance(self, attr, obj):
        '''
        Inputs
            attr: String of valid attribute
            obj: String of valid object
        Return
            Idx of a different attribute for the same object
        '''
        new_attr = np.random.choice(self.obj_affordance[obj])
        
        while new_attr == attr:
            new_attr = np.random.choice(self.obj_affordance[obj])
        
        return self.attr2idx[new_attr]

    def sample_train_affordance(self, attr, obj) -> int:
        '''
        Inputs
            attr: String of valid attribute
            obj: String of valid object
        Return
            Idx of a different attribute for the same object from the training pairs
        '''
        new_attr = np.random.choice(self.train_obj_affordance[obj])
        
        while new_attr == attr:
            new_attr = np.random.choice(self.train_obj_affordance[obj])
        
        return self.attr2idx[new_attr]

    def generate_features(self, out_file, model):
        '''
        Inputs
            out_file: Path to save features
            model: String of extraction model
        '''
        # data = self.all_data
        data = ospj(self.root, self.args.splitname)

        files_before = glob(ospj(data, '**', '*.jpg'), recursive=True)

        files_all = []
        for current in files_before:
            parts = current.split('/')
            if "cgqa" in self.root:
                files_all.append(parts[-1])
            else:
                files_all.append(os.path.join(parts[-2],parts[-1]))

        transform = dataset_transform('test', self.norm_family)
        feat_extractor = get_image_extractor(arch = model).eval()
        feat_extractor = feat_extractor.to(device)

        image_feats = []
        image_files = []
        for chunk in tqdm(
                chunks(files_all, 512), total=len(files_all) // 512, desc=f'Extracting features {model}'
            ):
            files = chunk
            imgs = list(map(self.loader, files))
            imgs = list(map(transform, imgs))
            feats = feat_extractor(torch.stack(imgs, 0).to(device))
            image_feats.append(feats.data.cpu())
            image_files += files

        image_feats = torch.cat(image_feats, 0)
        print('features for %d images generated' % (len(image_files)))

        torch.save({'features': image_feats, 'files': image_files}, out_file)

    def __getitem__(self, index):
        '''
        Call for getting samples
        '''
        index = self.sample_indices[index]

        image, attr, obj = self.data[index]

        # dbe(self.data[index])

        # Convert image to Phosc representation
        # image = Image.open(image_path)

        # if self.phosc_transorm:
        #     image = self.phosc_transorm(image)

        # If images need to be resized
        # image_resize = (self.args.image_resize_x, self.args.image_resize_y)
        # image.thumbnail(image_resize, Image.ANTIALIAS)

        # Decide what to output
        """
        if not self.update_features:
            img = self.activations[image]
        else:
            img = self.loader(image)
            img = self.transform(img)
        """
        # dbe(self.root, file_exists(ospj(self.root, self.split, image)))

        image_path = ospj(self.root, self.split, image)

        img = Image.open(image_path)

        trans = transforms.Compose([transforms.ToTensor()])
        img = trans(img)

        img = torch.tensor(img)
        img = img.unsqueeze(0)
        img = img.to(device)

        pred = self.phosc_model(img)

        pred_image = torch.cat([pred['phos'], pred['phoc']], dim=1) # torch.Size([1, 1395])

        d_attr = torch.tensor(self.clip_language_text).to(device)
        
        d_obj = gen_shape_description(obj)
        d_obj = clip.tokenize(d_obj)    # torch.size([13, 77])
        d_obj = torch.tensor(d_obj)

        # dbe(d_attr.shape, d_obj.shape)

        # FIX: Need to add the real values to what attr and obj should be also?
        # data = [d_image, d_attr, d_obj, self.pair2idx[(attr, obj)]]

        all_pairs = [self.pair2idx[pair] for pair in self.pairs]

        all_pairs = torch.tensor(all_pairs).to(device)

        data = {
            'image': {
                'path': image_path,
                'pred_image': pred_image,   # torch.Size([1, 1395])
                'pred_phos': pred['phos'],  # torch.Size([1, 195])
                'pred_phoc': pred['phoc']   # torch.Size([1, 1200])
            },
            'attr': {
                'pred': d_attr, # torch.Size([1, 77])
                'truth': attr,
                'truth_idx': self.attr2idx[attr],
                'clip': d_attr  # NOTE: Using d_attr here aswell in case I need something else for the value later
            },
            'obj': {
                'pred': d_obj,  # torch.Size([13, 77])
                'truth': obj,
                'truth_idx': self.obj2idx[obj],
                'clip': d_obj  # NOTE: Using d_obj here aswell in case I need something else for the value later
            },
            'pairs': {
                'pred': (d_attr, d_obj),
                'truth': (attr, obj),
                'truth_idx': self.pair2idx[(attr, obj)],
                'all': all_pairs
            }
        }

        if self.phase == 'train':
            all_neg_attrs = []
            all_neg_objs = []

            for curr in range(self.num_negs):
                neg_attr, neg_obj = self.sample_negative(attr, obj) # negative for triplet lose
                all_neg_attrs.append(neg_attr)
                all_neg_objs.append(neg_obj)

            neg_attr, neg_obj = torch.LongTensor(all_neg_attrs), torch.LongTensor(all_neg_objs)
            
            # dbe(self.train_obj_affordance)

            # NOTE: There probaly needs to be some changes here. This secion is only opperating on the index of the attr and obj, not the actual data.
            if len(self.train_obj_affordance[obj])>1:
                  inv_attr = self.sample_train_affordance(attr, obj) # attribute for inverse regularizer
            else:
                  inv_attr = (all_neg_attrs[0]) 

            comm_attr = self.sample_affordance(inv_attr, obj) # attribute for commutative regularizer

            data['attr']['neg'] = neg_attr
            data['attr']['inv'] = inv_attr
            data['attr']['comm'] = comm_attr

            data['obj']['neg'] = neg_obj

        return data
    
    def __len__(self):
        '''
        Call for length
        '''
        return len(self.sample_indices)
