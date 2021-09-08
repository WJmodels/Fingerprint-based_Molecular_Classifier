from chemprop.args import TrainArgs
from chemprop.args import PredictArgs
from chemprop.train import cross_validate,vs_predict_2,read_model_xgboost
from chemprop.utils import create_logger
import os
import pandas as pd
from chemprop.utils import load_checkpoint
from logging import Logger

if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = TrainArgs().parse_args()
    args.seed = 77#需要修改
    logger = create_logger(name='predict')
    #_,model = cross_validate(args, logger)
    # read_model_xgboost(args, logger)
    model = load_checkpoint(path=args.checkpoint_path, device=args.device, logger=logger)
    numbers,ranking_numbers = vs_predict_2(args, model,logger,'external_test/JAK2000.csv')
    print('numbers:',numbers)
    print('ranking_numbers',ranking_numbers)