# DMPNN_Morgan_XGBoost
A Powerful Approach to  Finding Novel Bruton Tyrosine Kinase Inhibitors

--train

python new_train.py --protein btk --dataset_type classification --data_path data/chembl_dmpnn_3000.csv --epochs 1 --hidden_size 150 --save_dir model_ymj 

--predict

python new_predict.py --checkpoint_path model_ymj/fold_0/model_0/model.pt --protein btk --dataset_type classification --data_path data/chembl_dmpnn_3000.csv 
