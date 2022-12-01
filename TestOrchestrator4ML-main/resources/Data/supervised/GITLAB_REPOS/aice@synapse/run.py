"""
Copyright (C) 2019  Syed Hasibur Rahman

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
# region : Imports
import os
import argparse
from synapse.utilities import load_module
from synapse.trainer.simpletrainer import SimpleTrainer
# endregion : Imports

# region : Globals
Config = None
# endregion : Globals


# region : Arguments
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("task",
                        type=str,
                        choices=['train', 'predict'],
                        help="Define task to run. Choices : [train, predict]")
    parser.add_argument("-c",
                        "--config-file",
                        help="Configuration file path",
                        default=os.path.join("projects", "mnist_example.py"),
                        required=True)
    parser.add_argument("-e",
                        "--epochs",
                        type=int,
                        help="Max epochs to train",
                        default=1,
                        required=True)

    return parser.parse_args()
# endregion : Arguments


def main():
    args = arg_parser()
    global Config
    print("Loading Configuration : ", args.config_file)
    Config = load_module(args.config_file).Config

    if args.task == 'train':
        trainer = SimpleTrainer(config=Config)
        trainer.train(epochs=args.epochs)


if __name__ == '__main__':
    main()
