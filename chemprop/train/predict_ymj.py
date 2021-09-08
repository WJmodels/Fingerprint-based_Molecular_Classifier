from logging import Logger
import os
from typing import Tuple
import pandas as pd
import numpy as np
import torch
import joblib

from .run_training import run_training,get_xgboost_feature
from .run_predicting_ymj import predict_feature
from chemprop.args import TrainArgs
from chemprop.data.utils import get_task_names
from chemprop.utils import makedirs
import xgboost as xgb
from sklearn.metrics import mean_squared_error,median_absolute_error
from sklearn.metrics import roc_curve, auc,precision_recall_curve,confusion_matrix,accuracy_score
from morgan_feature import get_morgan_feature
from chemprop.utils import load_checkpoint



def vs_predict_2(args,model,logger,external_test_path):
    #model = load_checkpoint(path=args.checkpoint_path, device=args.device, logger=logger)
    dmpnn_xgb = joblib.load('external_test/dmpnn_xgb.model')
    morgan_xgb = joblib.load('external_test/morgan_xgb.model')
    dmpnn_morgan_xgb = joblib.load('external_test/dmpnn_morgan_xgb.model')
    external_test_smiles,external_test_feature,external_test_preds,external_test_targets = predict_feature(args, logger, model, external_test_path)
    external_test_targets = [i[0] for i in external_test_targets]
    external_test_feature = pd.DataFrame(external_test_feature)
    external_test_morgan_feature = get_morgan_feature(external_test_smiles)
    external_test_gnn_mor_feature = pd.concat([external_test_feature, external_test_morgan_feature], axis=1)
    external_test_gnn_mor_feature.columns = range(external_test_gnn_mor_feature.shape[1])

    dmpnn_xgb_pre_pro = dmpnn_xgb.predict_proba(external_test_feature)[:,1]
    morgan_xgb_pre_pro = morgan_xgb.predict_proba(external_test_morgan_feature)[:,1]
    dmpnn_morgan_xgb_pre_pro = dmpnn_morgan_xgb.predict_proba(external_test_gnn_mor_feature)[:,1]

    input_data = pd.DataFrame([external_test_smiles,external_test_preds, dmpnn_xgb_pre_pro,morgan_xgb_pre_pro,dmpnn_morgan_xgb_pre_pro,external_test_targets]).T
    input_data.columns = ['test_smile', 'dmpnn_pre_pro','dmpnn_xgb_pre_pro', 'morgan_xgb_pre_pro', 'dmpnn_morgan_xgb_pre_pro','target']
    #input_data = input_data.sort_values(by='dmpnn_xgb_pre_pro', ascending=False)
    input_data.index = range(input_data.shape[0])
    numbers = int(input_data[input_data['dmpnn_xgb_pre_pro']>=0.9].shape[0])
    ranking_numbers = input_data[input_data['target'] == 1].index.tolist()
    input_data.to_csv('external_test/JAK2000_predict_150.csv',index=None)
    return numbers,ranking_numbers