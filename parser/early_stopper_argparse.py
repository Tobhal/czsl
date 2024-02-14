import argparse

def early_stopper_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    early_stopper_parser = parser.add_argument_group('early stopper arguments')

    early_stopper_parser.add_argument('--patience', type=int, default=7, help='patience for early stopping')
    early_stopper_parser.add_argument('--verbose', action='store_true', default=False, help='verbose for early stopping')
    early_stopper_parser.add_argument('--save_every', type=int, default=5, help='save model every n epochs')

    return parser