import argparse
from flags import DATA_FOLDER

def dataset_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    dataset_parser = parser.add_argument_group('Dataset arguments')

    dataset_parser.add_argument('--config', default='configs/args.yml', help='path of the config file (training only)')
    dataset_parser.add_argument('--dataset', default='mitstates', help='mitstates|zappos')
    dataset_parser.add_argument('--data_dir', default='mit-states', help='local path to data root dir from ' + DATA_FOLDER)
    dataset_parser.add_argument('--logpath', default=None, help='Path to dir where to logs are stored (test only)')
    dataset_parser.add_argument('--split_name', default='compositional-split-natural', help="dataset split")
    dataset_parser.add_argument('--cv_dir', default='logs/', help='dir to save checkpoints and logs to')
    dataset_parser.add_argument('--exp_name', default='temp', help='Name of exp used to name models')
    dataset_parser.add_argument('--load', default=None, help='path to checkpoint to load from')
    dataset_parser.add_argument('--image_extractor', default = 'resnet18', help = 'Feature extractor model')
    dataset_parser.add_argument('--norm_family', default = 'imagenet', help = 'Normalization values from dataset')
    dataset_parser.add_argument('--num_negs', type=int, default=1, help='Number of negatives to sample per positive (triplet loss)')
    dataset_parser.add_argument('--pair_dropout', type=float, default=0.0, help='Each epoch drop this fraction of train pairs')
    dataset_parser.add_argument('--test_set', default='val', help='val|test mode')
    dataset_parser.add_argument('--clean_only', action='store_true', default=False, help='use only clean subset of data (mitstates)')
    dataset_parser.add_argument('--subset', action='store_true', default=False, help='test on a 1000 image subset (debug purpose)')
    dataset_parser.add_argument('--open_world', action='store_true', default=False, help='perform open world experiment')
    dataset_parser.add_argument('--test_batch_size', type=int, default=32, help="Batch size at test/eval time")
    dataset_parser.add_argument('--cpu_eval', action='store_true', help='Perform test on cpu')
    dataset_parser.add_argument('--update_features', action='store_true', help='Update features')
    dataset_parser.add_argument('--train_only', action='store_true', help='Train only')
    dataset_parser.add_argument('--augmented', action='store_true', help='Augment data')

    return parser