import arl_fca_lab
import arl_binarization
import arl_data
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('.\\kras_binary\\result.csv', index_col=0, sep=',')
    # Исправление ошибки в номере электролизера было 50000, стало 5000, для всех подобных случаев
    df['Номер электролизера'] = pd.Series([i if i < 50000 else i - 45000 for i in df['Номер электролизера']])
    objt_list_5 = [i for i in df['Номер электролизера'].unique() if i < 6000]
    objt_list_6 = [i for i in df['Номер электролизера'].unique() if i >= 6000]
