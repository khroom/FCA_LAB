from collections import Set

import numpy
import unidecode
import numpy as np
import pandas as pd
import time
import networkx as nx
import matplotlib.pyplot as plt
import joblib
from networkx.drawing.nx_agraph import graphviz_layout
from fca_lab import fca_lattice
import arl_binarization
import pickle
import arl_data

class arl_fca_lab:
    def __init__(self, bin_type: arl_binarization.BinarizationType = arl_binarization.BinarizationType.HISTOGRAMS,
                 ind_type: bool = True, keep_nan=True, days_before_defect=1):
        """
        Конструктор класса. Инициализирует основные свойства.
        :param bin_type:
        :param ind_type:
        :param keep_nan:
        :param days_before_defect:
        """

        self.bin_type = bin_type
        self.ind_type = ind_type
        self.keep_nan=keep_nan
        self.days_before_defect = days_before_defect
        self.lat = None
        self.concepts_df = None
        self.confidence_df = {}
        self.defect = []
        # поставить условие на ind_type
        self.objs = []
        self.bin_matrix = arl_binarization.ArlBinaryMatrix()
        self.anomalies_only = False
        self.threshold = 0.34

    def fit_model(self, df: pd.DataFrame,  defect: str, obj_column: str, parse_date: str, anomalies_only: bool=False):
        """
        :param df:
        :param defect:
        :param obj_column:
        :param parse_dates:
        :return:
        """
        self.defect = defect
        if self.ind_type:
            self.objs = list(df.index.get_level_values(0).unique())
        else:
            self.objs = ['all']
        self.anomalies_only = anomalies_only

        self.bin_matrix.create_model(df, obj_column, parse_date, self.bin_type, defect, self.ind_type)

        print('Модель создана')
        self.bin_matrix.transform(df, self.defect, obj_column, parse_date, self.keep_nan,
                                                    self.days_before_defect, self.anomalies_only)
        # anomalies_only
        print('Данные бинаризованы')
        """
        Подготовка контекста для расчета решетки - pdf. Из исходного датасета убираем описательные столбцы,
        которые не содержат числовых параметров и относятся к описанию объектов исследования. Оставляем только
        строки у которых в столбце целевого параметра стоит 1.
        """
        for obj in self.objs:
            # Подумать как быть со служебными параметрами если они часть мультииндекса
            # И сделать для нескольких дефектов. Проверить как будет работать с [obj_column, parse_date]
            start_time = time.time()
            if not(obj == 'all'):
                pdf = self.bin_matrix.binary[
                    (self.bin_matrix.binary['1_day_before'] == 1) & (self.bin_matrix.binary[obj_column] == obj)].drop(
                    [obj_column, parse_date], axis='columns')

            else:
                pdf = self.bin_matrix.binary[(self.bin_matrix.binary['1_day_before'] == 1)].drop(
                    [obj_column, parse_date], axis='columns')
            # Инициализация решетки по урезанному контексту
            lat = fca_lattice(pdf)
            print("Объект ", obj, "\nКол-во нарушений:", len(pdf))
            lat.in_close(0, 0, 0)
            print(":\nгенерация концептов --- %s seconds ---" % (time.time() - start_time))

            # print("Кол-во концептов:", len(lat.concepts))

            """ 
            Датафрейм оценок концептов. 
            """
            start_time = time.time()
            self.confidence_df[obj] = self.rules_evaluation(lat, self.bin_matrix.binary, pdf)
            print("оценка концептов --- %s seconds ---" % (time.time() - start_time))

    def rules_evaluation(self, lat: fca_lattice, df: pd.DataFrame, pdf: pd.DataFrame):
        """
        Рассчет коэфициента уверенности для концептов относительно исходного контекста.
        :param lat:
        :param df:
        :return:
        """
        confidence_df = pd.DataFrame(index=range(len(lat.concepts)), columns=['B', 'confidence', 'd_support'])
        pdf_len = len(pdf)
        df_derivation = pd.Series(index=lat.context.columns, dtype='object')
        for col in df_derivation.index:
            df_derivation.loc[col] = set(df.loc[:, col][df.loc[:, col] == 1].index)

        for i in range(len(lat.concepts)):
            if (lat.concepts[i]['B'].difference({'1_day_before'}) != set()) and (lat.concepts[i]['A'] != set()):
                # Мощность содержания для левой части правила (без целевого параметра). Обязательно
                left_extent = len(
                    set.intersection(*[df_derivation[j] for j in lat.concepts[i]['B'].difference({'1_day_before'})]))
                # Можность содержания для всего правила (с целевым параметром). Обязательно
                rule_extent = len(lat.concepts[i]['A'])
                # Достоверность правила, как часто срабатывает. Обязательно
                confidence_df.loc[i, 'confidence'] = rule_extent / left_extent
                confidence_df.loc[i, 'd_support'] = rule_extent / pdf_len
                confidence_df.loc[i, 'B'] = lat.concepts[i]['B']

        confidence_df = confidence_df.sort_values(by='confidence', axis=0, ascending=False)
        return confidence_df



    def rules_scatter(self, obj_num, how: ['show', 'save'] = 'show', alpha=0.8, s=50):
        """
        ToDo: ДОбавить объект, убрать self.concepts_df
        Диаграмма рассеяния для весомых правил
        :param how: показывать или сохранить диаграмму в файл
        :param weight_df: Датафрейм оценок в разрезе ванн
        :param alpha: прозрачность
        :param s: размер маркеров
        :return:
        """
        fig, ax = plt.subplots()
        # fig = plt.figure(figsize=(9, 9))
        weight_df = self.confidence_df[obj_num].dropna()
        scatter = ax.scatter(x=weight_df['confidence'], y=weight_df['d_support'],
                             c=[len(weight_df.loc[i, 'B']) for i in weight_df.index],
                             cmap='viridis', alpha=alpha, s=s, label=weight_df['confidence'])
        legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Мощность правила")
        ax.add_artist(legend1)
        plt.xlabel("Уверенность")
        plt.ylabel("Поддержка")
        if how == 'show':
            plt.show()
        else:
            plt.savefig('scatter.png')

    def get_prediction(self, row_set: set, obj_num, support_threshold=0):
        # weight_df = self.concepts_df.sort_values(by='confidence', axis=0, ascending=False)
        for row_index in self.confidence_df[obj_num].index:
            if self.confidence_df[obj_num].loc[row_index, 'confidence'] >= self.threshold:
                rule_set = self.confidence_df[obj_num].loc[row_index, "B"]
                if isinstance(rule_set, Set):
                    if rule_set.difference(row_set) == {'1_day_before'}:
                        return self.confidence_df[obj_num].loc[row_index, 'confidence'], row_index
        return None


def get_row(df: pd.DataFrame, num: int = 0):
    row = df.iloc[num,2:-2]
    row_set = set()
    for (k, v) in row.items():
        if v:
            row_set.add(k)
    return row_set


def dump_model(model, file_name):
    picklefile = open(file_name, 'wb')
    pickle.dump(model, picklefile)
    picklefile.close()


def load_model(file_name):
    picklefile = open(file_name, 'rb')
    model = pickle.load(picklefile)
    picklefile.close()
    return model


if __name__ == '__main__':
    # matrix = new_model.bin_matrix.transform(dataset, new_model.defect, 'objt', 'dt', new_model.keep_nan, 0, new_model.anomalies_only)

    # df = pd.read_csv('.\\saz_binary\\Saz_histogramms_binary.csv', index_col=0, sep=',')
    print('Загрузка исходных данных')
    date_name = 'dt'
    obj_name = 'objt'
    defect_name = 'konus'
    dif_date = '2020.09.01'

    el_list = [0, 1, 2, 3, 4, 5, 6, 7]

    df = pd.read_csv('../FCA_LAB/result_19_01-20_11.csv', parse_dates=[date_name])

    # Предобработка датасета
    add_df = df['violation']
    df = df.drop(['violation','anod'], axis=1)
    # obj_restr = 9082
    train_df = df[(df[date_name] < dif_date)&(df[obj_name].isin(el_list))]
    # & (df['Номер электролизера'] == obj_restr)

    test_df = df[(df[date_name] >= dif_date)&(df[obj_name].isin(el_list))]
    # test_df = df[df['Дата'] >= '01.01.2020']

    print('Выполнено')

    print('Подготовка')

    train_df.set_index([obj_name, date_name], inplace=True)
    test_df.set_index([obj_name, date_name], inplace=True)

    test_df = test_df.drop(list(set(test_df.columns).difference(set(train_df.columns))), axis=1)
    print('Выполнено')

    print('Расчет модели')

    model = arl_fca_lab(bin_type=arl_binarization.BinarizationType.STDDEV, ind_type=False, keep_nan=True)
    model.fit_model(train_df, defect_name, obj_name, date_name, anomalies_only=True)

    # model = load_model('model_STD')
    # model.bin_matrix = r.ArlBinaryMatrix()
    # model.bin_matrix.create_model(train_df, 'Nomer elektrolizera', 'Data', model.bin_type, model.defect, model.ind_type,
    #                              model.days_before_defect)
    # model.rules_scatter(7.0, 'save')
    print('Выполнено')
    print('Подготовка тестового датасета')
    test_matrix = model.bin_matrix.transform(test_df, model.defect, obj_name, date_name,
                                             model.keep_nan, model.days_before_defect, model.anomalies_only)
    # dump_model(model,'model_QUAR')

    print('Выполнено')
    print('Расчет оценок')
    test_results = pd.DataFrame(index=test_matrix.index, columns=['Object', 'Prediction', 'Fact', 'Rule'])

    for i in test_matrix.index:

        if model.ind_type:
            obj_num = test_matrix.loc[i, obj_name]
        else:
            obj_num = 'all'
        rule_index = model.get_prediction(get_row(test_matrix, i), obj_num)
        test_results.loc[i, 'Object'] = test_matrix.loc[i, obj_name]
        test_results.loc[i, 'Fact'] = test_matrix.loc[i, '1_day_before']

        if rule_index == None:
            test_results.loc[i, 'Prediction'] = 0
            test_results.loc[i, 'Rule'] = -1
        else:
            test_results.loc[i, 'Prediction'] = model.confidence_df[obj_num].loc[rule_index, 'confidence']
            test_results.loc[i, 'Rule'] = rule_index
                # model.confidence_df[obj_num].loc[rule_index, 'B']
        print('Index:', i, ', Object:', test_results.loc[i, 'Object'], ', Predict:', test_results.loc[i, 'Prediction'],
              ', Fact:',test_results.loc[i, 'Fact'])
    print('Выполнено')
    test_results = pd.concat([test_results, test_matrix[[date_name, defect_name, '1_day_before']]], axis=1)

    # threshold_f1= pd.DataFrame(columns=[ 'Threshold'])
    # for threshold in numpy.arange(0.01, 0.99, 0.01):
    #     threshold_f1.loc[threshold, 'TP'] = len(
    #         test_results[(test_results.Fact == 1)&(test_results.Prediction >= threshold)])
    #     threshold_f1.loc[threshold, 'FP'] = len(
    #         test_results[(test_results.Fact == 0)&(test_results.Prediction >= threshold)])
    #     threshold_f1.loc[threshold, 'FN'] = len(
    #         test_results[(test_results.Fact == 1) & (test_results.Prediction < threshold)])
    #     threshold_f1.loc[threshold, 'TN'] = len(
    #         test_results[(test_results.Fact == 0) & (test_results.Prediction < threshold)])
    #     threshold_f1.loc[threshold, 'TN_TP'] = threshold_f1.loc[threshold, 'TN']+threshold_f1.loc[threshold, 'TP']
    #     threshold_f1.loc[threshold, 'Threshold'] = threshold
    #
    # fig, ax = plt.subplots()
    # ax.plot(threshold_f1['Threshold'], threshold_f1['TP'] / len(test_results[test_results['Fact'] == 1]), 'g',
    #         threshold_f1['Threshold'], threshold_f1['FP'] / len(test_results[(test_results[defect_name] == 0)&(test_results['Fact'] == 0)]), 'r')
    #
    # # plt.legend(['Истинно положительных', 'Ложно положительных'])
    # plt.xlabel("Пороговое значение")
    # plt.ylabel("Доля")
    # # plt.title('Vanna-9001, bin_type=STDDEV, ind_type=True, \nkeep_nan=True, anomalies_only=False, \n'
    # #           'train(2018.01.01-2019.01.01), test(2019.01.01-2019.08.01)')
    # plt.legend(['Истинно положительные', 'Ложно положительные'])
    # plt.savefig('temp_std.png')
    # plt.clf()
    #
    # fig, ax = plt.subplots()
    # ax.bar(test_results.index, test_results[test_results.Object == 0]['Prediction'], color=plt.cm.Paired(1), label='Prediction')
    # ax.bar(test_results.index, test_results['Fact'], color=plt.cm.Paired(7), alpha=0.6, label='Fact')
    # ax.legend()
    # plt.xlabel("Порядковый индекс")
    # plt.ylabel("Коэффициент уверенности")
    # plt.savefig('temp_bar_std.png')
    #
    # sr = (threshold_f1['TN_TP'] / len(test_results))
    # print(threshold_f1[sr==sr.max()])
    # print(len(test_results[(test_results[defect_name] == 1) & (test_results.Prediction >= 0.61) & (test_results.Fact == 0)]))
