"""Trains a model on a dataset."""

from chemprop.args import TrainArgs
from chemprop.train import cross_validate,vs_predict,read_model_xgboost
from chemprop.utils import create_logger
import os
import pandas as pd

if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = TrainArgs().parse_args()
    args.seed = 77#需要修改
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
    _,model = cross_validate(args, logger)
    # read_model_xgboost(args, logger)
    #numbers,ranking_numbers = vs_predict(args, model,logger,'external_test/extra_vs2.csv')
    #print('numbers:',numbers)
    #print('ranking_numbers',ranking_numbers)
    vs_predict(args, model,logger,'external_test/extra_vs_24_3.csv')

