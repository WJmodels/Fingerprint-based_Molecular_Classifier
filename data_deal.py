import pandas as pd
import numpy as np



def btk_data_decoy_old():
    df = pd.read_csv('btk_active_decoy/BTK_2810_old.csv')
    df_decoy = pd.read_csv('btk_active_decoy/btk_finddecoy.csv')
    df_decoy = pd.DataFrame(df_decoy['smile'])
    df_decoy['label'] = 0
    df_active = df[df['target2']<300]
    df_active['target2'] = 1
    df_ic_decoy = df[df['target2']>9000]
    df_ic_decoy['target2'] = 0
    del df_active['target1'],df_ic_decoy['target1']
    df_active.columns = df_ic_decoy.columns = df_decoy.columns = ['smiles','label']
    df_all = pd.concat([df_active,df_ic_decoy])
    df_all = pd.concat([df_all,df_decoy])
    df_all.to_csv('btk_active_decoy/btk_2810_add_decoy_old.csv',index=None)

def btk_data_cut_decoy():
    df = pd.read_csv('btk_active_decoy/BTK_2810_old.csv')
    df_decoy = pd.read_csv('btk_active_decoy/btk_finddecoy.csv')
    df_cut_decoy = pd.read_csv('btk_active_decoy/similarity_active_decoy.csv')
    df_cut_decoy = df_cut_decoy.head(1139)#1139是根据正样本1393个，乘以10比例13930，原先数据decoy总量为15069,15069-13930=1139
    df_decoy = pd.DataFrame(df_decoy['smile'])
    df_decoy['label'] = 0
    df_active = df[df['target2']<300]
    df_active['target2'] = 1
    df_ic_decoy = df[df['target2']>9000]
    df_ic_decoy['target2'] = 0
    del df_active['target1'],df_ic_decoy['target1']
    df_active.columns = df_ic_decoy.columns = df_decoy.columns = ['smiles','label']
    df_all = pd.concat([df_active,df_ic_decoy])
    df_all = pd.concat([df_all,df_decoy])
    df_all_filter = df_all[~ df_all['smiles'].isin(df_cut_decoy['train_smiles'])]
    df_all_filter.to_csv('btk_active_decoy/btk_2810_cut_decoy.csv',index=None)


def btk_data_decoy():
    df = pd.read_csv('btk_active_decoy/BTK_2810.csv')
    df_decoy = pd.read_csv('btk_active_decoy/btk_finddecoy.csv')
    df_decoy = df_decoy.sample(frac=0.5, random_state=123)
    df_decoy = pd.DataFrame(df_decoy['smile'])
    df_decoy['label'] = 0
    df_active = df[df['target2']<300]
    df_active['target2'] = 1
    df_ic_decoy = df[df['target2']>9000]
    df_ic_decoy['target2'] = 0
    del df_active['target1'],df_ic_decoy['target1']
    df_active.columns = df_ic_decoy.columns = df_decoy.columns = ['smiles','label']
    df_all = pd.concat([df_active,df_ic_decoy])
    df_all = pd.concat([df_all,df_decoy])
    df_all.to_csv('btk_active_decoy/btk_2810_add_decoy.csv',index=None)


def btk_2810_ic50():
    df = pd.read_csv('btk_active_decoy/BTK_2810_old.csv')
    df_decoy = pd.read_csv('btk_active_decoy/btk_finddecoy.csv')
    df_decoy = pd.DataFrame(df_decoy['smile'])
    df_decoy['label'] = 2
    df_active = df[df['target2']<300]
    df_ic_decoy = df[df['target2']>9000]
    del df_active['target1'],df_ic_decoy['target1']
    df_active.columns = df_ic_decoy.columns = df_decoy.columns = ['smiles','label']
    df_all = pd.concat([df_active,df_ic_decoy])
    df_all = pd.concat([df_all,df_decoy])
    df_all.to_csv('btk_active_decoy/btk_2810_ic50.csv',index=None)


def btk_2610_data():
    df = pd.read_csv('btk_active_decoy/BTK_2610.csv')
    df_decoy = pd.read_csv('btk_active_decoy/btk_2610_find_decoy.csv')
    df_decoy = pd.DataFrame(df_decoy['smile'])
    df_decoy['label'] = 0
    df_active = df[df['ic50']<100]
    df_active['ic50'] = 1
    df_ic_decoy = df[df['ic50']>=1000]
    df_ic_decoy['ic50'] = 0
    del df_active['chemblid'],df_ic_decoy['chemblid']
    df_active.columns = df_ic_decoy.columns = df_decoy.columns = ['smiles','label']
    df_all = pd.concat([df_active,df_ic_decoy])
    df_all = pd.concat([df_all,df_decoy])
    df_all.to_csv('btk_active_decoy/btk_2610_add_decoy.csv',index=None)

def btk_our_data():
    df = pd.read_csv('btk_active_decoy/chembl_train_add_decoy.csv')
    df_decoy = pd.read_csv('btk_active_decoy/btk_our_decoy.csv')
    df_decoy = pd.DataFrame(df_decoy['smile'])
    df_decoy = df_decoy.sample(frac=0.2, random_state=123)
    df_decoy['label'] = 0
    df.columns = df_decoy.columns = ['smiles','label']
    df_all = pd.concat([df,df_decoy])
    df_all.to_csv('btk_active_decoy/btk_our_add_decoy.csv',index=None)

if __name__ == '__main__':
    btk_data_cut_decoy()


