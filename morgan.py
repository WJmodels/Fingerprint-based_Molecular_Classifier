from rdkit.Chem import Descriptors, Lipinski, Crippen
from rdkit.Chem import AllChem as chem
import os
import pandas as pd
from rdkit import DataStructs


def get_simialarity(path1,path2,path3):
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df1 = df1[df1['label']==1]
    result = []
    for i in df1['smiles']:
        active_fp = chem.GetMorganFingerprint(chem.MolFromSmiles(i), 2)
        one_similarity = []
        for k,j in enumerate(df2['smiles']):
            decoy_fp = chem.GetMorganFingerprint(chem.MolFromSmiles(j), 2)
            one_similarity.append([i,j,DataStructs.TanimotoSimilarity(active_fp, decoy_fp),df2['label'][k]])
        for y in one_similarity:
            result.append(y)
    result_df = pd.DataFrame(result)
    result_df.columns = ['begin_smiles','train_smiles','similarity_scores','train_ic50']
    result_df.to_csv(path3,index=None)

def run_data():
    df = pd.read_csv('data/chembl_btk.csv')
    df = df[['Smiles','Standard Type','Standard Relation','Standard Value']]
    # df = df[(df['Standard Type'] == 'IC50') | (df['Standard Type'] == 'Ki') | (df['Standard Type'] == 'Kd') | (
    #             df['Standard Type'] == 'Kd apparent')]
    df = df[df['Standard Type'] == 'IC50']
    # df = df[df['ASSAY_ORGANISM'] == 'Homo sapiens']
    df = df.dropna(axis=0, how='any')
    df_active = df[(df['Standard Relation'] != '>') & (df['Standard Relation'] != '>=')]
    df_active = df_active[['Smiles','Standard Value']]
    df_active['Standard Value'] = df_active['Standard Value'].astype('float')
    df_active.columns = ['smiles','ic50']
    df_active.index = range(df_active.shape[0])
    df_active.to_csv('data/chembl_btk_deal.csv',index=None)

def concat_data():
    chembl_data = pd.read_csv('data/chembl_btk_deal.csv')
    zinc_data = pd.read_csv('data/zinc_ic50.csv')
    concat_data_df = pd.concat([chembl_data,zinc_data],axis=0)
    concat_data_df = concat_data_df.drop_duplicates(['smiles'],keep='first')
    concat_data_df.to_csv('data/btk_zinc_chembl.csv',index=None)

def tet():
    df = pd.read_csv('/home/cxw/下载/SubStr(1).csv')
    for j,i in enumerate(df['Smiles']):
        print(chem.MolFromSmiles(i))
        print(j)

def split_data_new(path):
    df = pd.read_csv(path)
    df_active = df[df['ic50']<30]
    df_decoy = df[df['ic50']>500]
    df_active['ic50'] = 1
    df_decoy['ic50'] = 0
    df = pd.concat([df_active,df_decoy])
    df.to_csv('data/chembl_data_30_500.csv',index=None)

def concat_decoy():
    df = pd.read_csv('data/chembl_btk.csv')
    df = df[['Smiles','Comment']]
    df = df[(df['Comment'] == 'inactive') | (df['Comment'] == 'Not Active')]
    df['Comment'] = 0
    df.columns = ['smiles','ic50']
    df.to_csv('data/inactive.csv',index=None)
    df_ic50 = pd.read_csv('data/chembl_data_30_500.csv')
    df_result = pd.concat([df_ic50,df])
    df_result.to_csv('data/ic50_add_inactive.csv',index=None)

def concat_data():
    df_inactive = pd.read_csv('data/inactive.csv')
    df_chembl = pd.read_csv('data/btk_zinc_chembl.csv')
    df_ic50_decoy = df_chembl[df_chembl['ic50']>=100]
    df_inactive.columns = df_ic50_decoy.columns = ['smiles','label']
    df_inactive['label'] = 0
    df_ic50_decoy['label'] = 0
    df = pd.read_csv('data/chembl_train.csv')
    df_concat = pd.concat([df,df_inactive])
    df_concat = pd.concat([df_concat,df_ic50_decoy])
    df_concat = df_concat.drop_duplicates(subset=['smiles'], keep='first', inplace=False)
    df_concat.to_csv('data/chembl_train_add_decoy.csv',index=None)


if __name__=="__main__":
    # tet()
    # run_data()
    # concat_data()
    # split_data_new('data/btk_zinc_chembl.csv')
    # concat_data()
    get_simialarity('external_test/extra_vs.csv', 'btk_active_decoy/btk_2810_ic50.csv', 'external_test/similarity.csv')