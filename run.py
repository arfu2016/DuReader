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


# ssh 192.168.10.2
# conda info --env
# source activate dureader3
# export PATH=/home/deco/local/cuda-8.0/bin${PATH:+:${PATH}}
# export LD_LIBRARY_PATH=/home/deco/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
# nvcc --version
# nvida-smi
# cd projects/DuReader
# python run.py --predict --algo MLSTM --batch_size 16 --gpu "0,1" --test_files '../data/demo/testset/search.test.json'
# python run.py --evaluate --algo MLSTM --batch_size 16
# python run.py --train --algo MLSTM --epochs 2 --batch_size 16 --gpu "0,1"

# python run.py --predict --algo MLSTM --batch_size 16 --test_files '../data/demo/testset/search.test.json'

# python run.py --train --algo MLSTM --epochs 2 --batch_size 16
# python mreading/run_python3.py --train --algo MLSTM --epochs 2 --batch_size 16

# python mreading/run_python3.py --prepare
# --train_files '../data/preprocessed/trainset/search.train.json'
# --dev_files '../data/preprocessed/devset/search.dev.json'
# --test_files '../data/demo/testset/search.test.json'

# python mreading/run_python3.py --train --algo MLSTM --epochs 2 --batch_size 16
# --train_files '../data/preprocessed/trainset/search.train.json'
# --dev_files '../data/preprocessed/devset/search.dev.json'

# python squad2/run_python3.py --prepare
# python squad2/run_python3.py --train --algo MLSTM --epochs 1 --batch_size 16
# python squad2/run_python3.py --train --algo MLSTM --epochs 1 --batch_size 32
# python squad2/run_python3.py --predict --algo MLSTM --epochs 1 --batch_size 32

# python squad2/run_python3.py --prepare
# --train_files '/home/projects/DuReader/data/demo_squad/train_demo.json'
# --dev_files '/home/projects/DuReader/data/demo_squad/valid_demo.json'
# --test_files '/home/projects/DuReader/data/demo_squad/valid_demo.json'

# python squad2/run_python3.py --train --algo MLSTM --epochs 1 --batch_size 32
# --train_files '/home/projects/DuReader/data/demo_squad/train_demo.json'
# --dev_files '/home/projects/DuReader/data/demo_squad/valid_demo.json'
# --test_files '/home/projects/DuReader/data/demo_squad/valid_demo.json'

# python squad2/run_python3.py --predict --algo MLSTM  --epochs 1 --batch_size 32
# --train_files '/home/projects/DuReader/data/demo_squad/train_demo.json'
# --dev_files '/home/projects/DuReader/data/demo_squad/valid_demo.json'
# --test_files '/home/projects/DuReader/data/demo_squad/valid_demo.json'

# python squad_mlstm/run_python3.py --prepare
# python squad_mlstm/run_python3.py --train --algo MLSTM --epochs 1 --batch_size 32
# python squad_mlstm/run_python3.py --predict --algo MLSTM --epochs 1 --batch_size 32

# python squad_mlstm/run_python3.py --train --algo BIDAF --epochs 1 --batch_size 32
# python squad_mlstm/run_python3.py --predict --algo BIDAF --epochs 1 --batch_size 32
