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

    def fit_model(self, df: pd.DataFrame,  defect: str, obj_column: str, parse_date: str):
        """
        :param df:
        :param defect:
        :param obj_column:
        :param parse_dates:
        :return:
        """
        self.defect = defect
        self.objs = list(df.index.get_level_values(0).unique())
        # self.objs = [5.0]

        self.bin_matrix.create_model(df, obj_column, parse_date, self.bin_type, defect, self.ind_type)

        print('Модель создана')
        self.bin_matrix = self.bin_matrix.transform(df, self.defect, obj_column, parse_date, self.keep_nan,
                                                    self.days_before_defect)
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
            pdf = self.bin_matrix[(self.bin_matrix['1_day_before'] == 1) & (self.bin_matrix[obj_column] == obj)].drop(
                [obj_column, parse_date],
                axis='columns')
            # Инициализация решетки по урезанному контексту
            lat = fca_lattice(pdf)
            print("Объект ", obj, "\nКол-во нарушений:", len(pdf))
            lat.in_close(0, 0, 0)
            print(":\nгенерация концептов --- %s seconds ---" % (time.time() - start_time))

            # print("Кол-во концептов:", len(lat.concepts))

            """ 
            Датафрейм оценок концептов. 
            Можно было бы совместить с self.lat.concepts, но это список и он во вложеном объекте.
            Важно формировать после расчета концептов.
            """
            # self.concepts_df = pd.DataFrame(index=range(len(self.lat.concepts)),
            #                                 columns=['param_support', 'support', 'confidence', 'lift', 'cardinality', 'intent'])

            start_time = time.time()
            self.confidence_df[obj] = self.rules_evaluation(lat, self.bin_matrix, pdf)
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


    def rules_describe(self, weight_df: pd.DataFrame):
        """
        ToDo: Возможно пригодиться
        Описание правил формальным образом для генерации отчета по проекту. Вариант.
        :param weight_df: Датафрейм оценок в разрезе ванн
        :return:
        """
        df = weight_df.groupby(by='VANNA', axis=0).max()
        for i in weight_df.iloc[:, 2:].columns:
            print('Правило №', i)
            print(self.concepts[i]['B'].difference({self.param}), '--> {\'', self.param, '\'}')
            print('Мощность:', len(self.concepts[i]['A']))
            # print('Частичная поддержка: ', self.concepts[i]['part_support'])
            # print('Полная поддержка: ', self.concepts[i]['full_support'])
            print('Вес: ', np.around(weight_df[i].max(), decimals=2))
            print('Количество ванн:', df[i].count())
            print('---')

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
            if self.confidence_df[obj_num].loc[row_index, 'd_support'] >=support_threshold:
                rule_set = self.confidence_df[obj_num].loc[row_index, "B"]
                if isinstance(rule_set, Set):
                    if rule_set.difference(row_set) == {'1_day_before'}:
                        return row_index
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
    # df = pd.read_csv('.\\saz_binary\\CAZ_binary_stdev_individual_true.csv', index_col=0, sep=',')
    # param = 'Konus (da/net)'

    # df = pd.read_csv('.\\saz_binary\\Saz_histogramms_binary.csv', index_col=0, sep=',')
    print('Загрузка исходных данных')
    df = pd.read_csv('.\\resultall.csv', parse_dates=['Дата'])

    test_df = df[(df['Дата'] >= '2020.02.01')&(df['Дата'] < '2020.05.01')]
    # test_df = df[df['Дата'] >= '01.01.2020']
    train_df = df[(df['Дата'] >= '2019.01.01')&(df['Дата'] < '2020.01.01')]
    print('Выполнено')
    print('Переход на латиницу')
    test_df.drop(['Индекс', 'АЭ: кол-во', 'АЭ: длит.', 'АЭ: напр. ср.', 'Пена: снято с эл-ра', 'Срок службы',
                  'Кол-во поддер. анодов', 'АЭ: кол-во локальных', 'Отставание (шт)', 'Другое нарушение (шт)',
                  'Нарушение', 'Анод', 'Выливка: отношение скорости', 'Выход по току: Л/О', 'Подина: напр.',
                  'Кол-во доз АПГ в руч.реж.', 'Конус (да/нет)'], axis='columns', inplace=True)

    # замена кириллицы на латинницу
    for column in test_df.columns:
        test_df.rename(columns={column: unidecode.unidecode(column).replace('\'', '')}, inplace=True)
    test_df = test_df.set_index(['Nomer elektrolizera', 'Data'])

    train_df.drop(['Индекс', 'АЭ: кол-во', 'АЭ: длит.', 'АЭ: напр. ср.', 'Пена: снято с эл-ра', 'Срок службы',
                  'Кол-во поддер. анодов', 'АЭ: кол-во локальных', 'Отставание (шт)', 'Другое нарушение (шт)',
                  'Нарушение', 'Анод', 'Выливка: отношение скорости', 'Выход по току: Л/О', 'Подина: напр.',
                  'Кол-во доз АПГ в руч.реж.', 'Конус (да/нет)'], axis='columns', inplace=True)

    # замена кириллицы на латинницу
    for column in train_df.columns:
        train_df.rename(columns={column: unidecode.unidecode(column).replace('\'', '')}, inplace=True)
    train_df = train_df.set_index(['Nomer elektrolizera', 'Data'])
    print('Выполнено')
    print('Расчет модели')

    # model = arl_fca_lab(bin_type=arl_binarization.BinarizationType.QUARTILES)
    # model.fit_model(train_df,'Konus (sht)','Nomer elektrolizera', 'Data')

    model = load_model('model_QUAR')
    model.bin_matrix = arl_binarization.ArlBinaryMatrix()
    model.bin_matrix.create_model(train_df, 'Nomer elektrolizera', 'Data', model.bin_type, model.defect, model.ind_type,
                                 model.days_before_defect)
    # model.rules_scatter(7.0, 'save')
    print('Выполнено')
    print('Подготовка тестового датасета')
    test_matrix = model.bin_matrix.transform(test_df, model.defect, 'Nomer elektrolizera', 'Data',
                                             model.keep_nan, model.days_before_defect)
    # dump_model(model,'model_QUAR')

    print('Выполнено')
    print('Расчет оценок')
    test_results = pd.DataFrame(index=test_matrix.index, columns=['Object', 'Prediction', 'Fact', 'Rule'])

    for i in test_matrix.index:
        obj_num = test_matrix.loc[i, 'Nomer elektrolizera']
        rule_index = model.get_prediction(get_row(test_matrix, i), obj_num)
        test_results.loc[i, 'Object'] = obj_num
        test_results.loc[i, 'Fact'] = test_matrix.iloc[i, -1]

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
    test_results = pd.concat([test_results, test_matrix[['Data', 'Konus (sht)', '1_day_before']]], axis=1)
    threshold_f1= pd.DataFrame(columns=[ 'Threshold', 'precision', 'recall'])
    for threshold in numpy.arange(0.01, 0.99, 0.01):
        threshold_f1.loc[threshold, 'TP'] = len(
            test_results[(test_results.Fact == 1) &
                         (test_results.Prediction >= threshold)])+ len(
            test_results[(test_results['Konus (sht)'] != 0) &
                         (test_results.Prediction >= threshold)])

        threshold_f1.loc[threshold, 'FP'] = len(
            test_results[(test_results.Fact == 0) &
                         (test_results.Prediction >= threshold) &
                         (test_results['Konus (sht)'] == 0)])
        threshold_f1.loc[threshold, 'FN'] = len(
            test_results[(test_results.Fact == 1) & (test_results.Prediction < threshold)])
        threshold_f1.loc[threshold, 'Threshold'] = threshold
        threshold_f1.loc[threshold, 'recall'] = threshold_f1.loc[threshold, 'TP'] / (
                    threshold_f1.loc[threshold, 'TP'] + threshold_f1.loc[threshold, 'FN'])
        threshold_f1.loc[threshold, 'precision'] = threshold_f1.loc[threshold, 'TP'] / (
                    threshold_f1.loc[threshold, 'TP'] + threshold_f1.loc[threshold, 'FP'])
        # threshold_f1.loc[threshold, 'F1'] = 2 / (1/threshold_f1.loc[threshold, 'recall'] +
        #                                          1/threshold_f1.loc[threshold, 'precision'])

    plt.plot(threshold_f1['Threshold'], threshold_f1['TP'] / len(test_results[(test_results['Konus (sht)'] == 1)|
                                                                         (test_results['1_day_before'] == 1)]), 'r',
             threshold_f1['Threshold'], threshold_f1['FP'] / len(test_results[(test_results['Konus (sht)'] == 0)&
                                                                         (test_results['1_day_before'] == 0)]), 'b',
             threshold_f1['Threshold'], threshold_f1['FN'] / len(test_results[(test_results['Konus (sht)'] == 1)|
                                                                         (test_results['1_day_before'] == 1)]), 'g')
    plt.savefig('quar_02-05_1.png')

