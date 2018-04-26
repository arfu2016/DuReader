"""
@Project   : DuReader
@Module    : shortcuts.py
@Author    : Deco [deco@cubee.com]
@Created   : 4/9/18 4:18 PM
@Desc      : The shortcut of entrance of the project
"""
import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from tensorflow2.run_python3 import run

if __name__ == '__main__':
    run()

# python run.py --train --algo MLSTM --epochs 2 --batch_size 16 --gpu "0,1" \
# --train_files '../data/preprocessed/trainset/search.train.json' \
# --dev_files '../data/preprocessed/devset/search.dev.json'

# python run.py --predict --algo MLSTM --batch_size 16 --gpu "0,1" \
# --test_files '../data/preprocessed/testset/search.test.json'

# python run.py --predict --algo MLSTM --batch_size 16 --gpu "0,1" \
# --test_files '../data/demo/testset/search.test.json'

# python run.py --prepare \
# --train_files '../data/preprocessed/trainset/search.train.json' \
# --dev_files '../data/preprocessed/devset/search.dev.json' \
# --test_files '../data/demo/testset/search.test.json'

# python run.py --prepare \
# --train_files '../data/preprocessed/trainset/search.train.json' \
# --dev_files '../data/preprocessed/devset/search.dev.json' \
# --test_files '../data/preprocessed/testset/search.test.json'

# python run.py --train --algo MLSTM --epochs 2 --batch_size 16 --gpu "0,1"
# python run.py --evaluate --algo MLSTM --batch_size 16
