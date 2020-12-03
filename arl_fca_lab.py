import numpy as np
import pandas as pd
import time
import networkx as nx
import matplotlib.pyplot as plt
import joblib
from networkx.drawing.nx_agraph import graphviz_layout
from fca_lab import fca_lattice
import arl_binarization


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

        # self.objs = [1.0, 2.0]

        # Определить параметры функции !!!!!!
        self.bin_matrix.create_model(df, obj_column, parse_date, self.bin_type, defect, self.ind_type,
                                     self.days_before_defect)
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
            self.confidence_df[obj] = self.rules_evaluation(lat, self.bin_matrix)
            print("оценка концептов --- %s seconds ---" % (time.time() - start_time))

    def rules_evaluation(self, lat: fca_lattice, df: pd.DataFrame):
        """
        Рассчет коэфициента уверенности для концептов относительно исходного контекста.
        :param lat:
        :param df:
        :return:
        """
        confidence_df = pd.DataFrame(index=range(len(lat.concepts)), columns=['B', 'confidence'])

        df_derivation = pd.Series(index=lat.context.columns, dtype='object')
        for col in df_derivation.index:
            df_derivation.loc[col] = set(df.loc[:, col][df.loc[:, col] == 1].index)

        for i in range(len(lat.concepts)):
            if (lat.concepts[i]['B'].difference({'1_day_before'}) != set()) and (lat.concepts[i]['A'] != set()):
                # Мощность содержания для левой части правила (без целевого параметра). Обязательно
                left_extent = len(
                    set.intersection(*[df_derivation[j] for j in lat.concepts[i]['B'].difference({self.defect[0]})]))
                # Можность содержания для всего правила (с целевым параметром). Обязательно
                rule_extent = len(lat.concepts[i]['A'])
                # Достоверность правила, как часто срабатывает. Обязательно
                confidence_df.loc[i, 'confidence'] = rule_extent / left_extent
                confidence_df.loc[i, 'B'] = lat.concepts[i]['B']
        confidence_df = confidence_df.sort_values(by='confidence', axis=0, ascending=False)
        return confidence_df

    def concepts_evaluation(self, df, pdf):
        """
        Рассчет параметров для концептов относительно исходного контекста.
        :return:
        """
        df_derivation = pd.Series(index=self.lat.context.columns, dtype='object')
        for col in df_derivation.index:
            df_derivation.loc[col] = set(df.loc[:, col][df.loc[:, col] == 1].index)
        # param_fraction = len(df[df[self.defect[0]] == 1]) / len(df)

        # df_len = len(df)
        # pdf_len = len(pdf)

        for i in range(len(self.lat.concepts)):
            if (self.lat.concepts[i]['B'].difference({self.defect[0]}) != set()) and (self.lat.concepts[i]['A'] != set()):
                # Мощность содержания для левой части правила (без целевого параметра). Обязательно
                self.lat.concepts[i]['left_extent'] = len(
                    set.intersection(*[df_derivation[j] for j in self.lat.concepts[i]['B'].difference({self.defect[0]})]))
                # Можность содержания для всего правила (с целевым параметром). Обязательно
                self.lat.concepts[i]['rule_extent'] = len(self.lat.concepts[i]['A'])
                # Поддержка по нарушениям. Опционально
                # self.concepts_df.loc[i, 'param_support'] = self.lat.concepts[i]['rule_extent'] / pdf_len
                # Поддержка правила. Опционально
                # self.concepts_df.loc[i, 'support'] = self.lat.concepts[i]['rule_extent'] / df_len
                # Мощность содержания левой части правила. Опционально
                # self.concepts_df.loc[i, 'cardinality'] = len(self.lat.concepts[i]['B']) - 1
                # Достоверность правила, как часто срабатывает. Обязательно
                self.concepts_df.loc[i, 'confidence'] = self.lat.concepts[i]['rule_extent'] / self.lat.concepts[i]['left_extent']
                # Поддержка, вклад левой части в вероятность правой части. Опционально
                # self.concepts_df.loc[i, 'lift'] = self.concepts_df.loc[i, 'confidence'] / param_fraction
                # Количество нарушений покрытых правилом. Опционально
                # self.concepts_df.loc[i, 'extent'] = len(self.lat.concepts[i]['A'])

    def rules_describe(self, weight_df: pd.DataFrame):
        """
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

    def rules_scatter(self, how: ['show', 'save'] = 'show', alpha=0.8, s=50):
        """
        Диаграмма рассеяния для весомых правил
        :param how: показывать или сохранить диаграмму в файл
        :param weight_df: Датафрейм оценок в разрезе ванн
        :param alpha: прозрачность
        :param s: размер маркеров
        :return:
        """
        fig, ax = plt.subplots()
        # fig = plt.figure(figsize=(9, 9))
        weight_df = self.concepts_df.sort_values(by='cardinality', axis=0)
        scatter = ax.scatter(x=weight_df['confidence'], y=weight_df['extent'], c=weight_df['cardinality'],
                             cmap='viridis', alpha=alpha, s=s, label=weight_df['cardinality'])
        legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="cardinality")
        ax.add_artist(legend1)
        plt.xlabel("Confidence")
        plt.ylabel("Extent")
        if how == 'show':
            plt.show()
        else:
            plt.savefig('scatter.png')

    def get_prediction(self, row_set: set):
        weight_df = self.concepts_df.sort_values(by='confidence', axis=0, ascending=False)
        for i in weight_df.index:
            if self.lat.concepts[i]['B'].difference(row_set) == {self.param}:
                return weight_df.loc[i,'confidence']
                break
        return 0

def get_row(df: pd.DataFrame, num: int = 0):
    row = df.iloc[num,2:-2]
    row_set = set()
    for (k, v) in row.items():
        if v:
            row_set.add(k)
    return row_set

if __name__ == '__main__':
    # df = pd.read_csv('.\\saz_binary\\CAZ_binary_stdev_individual_true.csv', index_col=0, sep=',')
    # param = 'Konus (da/net)'

    # df = pd.read_csv('.\\saz_binary\\Saz_histogramms_binary.csv', index_col=0, sep=',')
    df = pd.read_csv('.\\resultall.csv', parse_dates=['Дата'])
    df = df[df['Дата'] < '01.01.2020']

    df.drop(['Индекс', 'АЭ: кол-во', 'АЭ: длит.', 'АЭ: напр. ср.', 'Пена: снято с эл-ра', 'Срок службы',
                  'Кол-во поддер. анодов', 'АЭ: кол-во локальных', 'Отставание (шт)', 'Другое нарушение (шт)',
                  'Нарушение', 'Анод', 'Выливка: отношение скорости', 'Выход по току: Л/О', 'Подина: напр.',
                  'Кол-во доз АПГ в руч.реж.', 'Конус (да/нет)'], axis='columns', inplace=True)
    import unidecode

    # замена кириллицы на латинницу
    for column in df.columns:
        df.rename(columns={column: unidecode.unidecode(column).replace('\'', '')}, inplace=True)
    df = df.set_index(['Nomer elektrolizera', 'Data'])

    model = arl_fca_lab()
    model.fit_model(df,'Konus (sht)','Nomer elektrolizera', 'Data')


    # df = df.drop(columns='Konus (da/net)')
    # param = 'DAY_BEFORE_KONUS'

    # binary = df[df['VANNA'] == 0]
    # arl = arl_fca_lab(binary, ['VANNA', 'RDATE'], param)
    # print("Загрузка --- %s seconds ---" % (time.time() - start_time))
    # start_time = time.time()
    # arl.lat.in_close(0, 0, 0)
    # print("Генерация концептов --- %s seconds ---" % (time.time() - start_time))
    # # arl.lat.concepts.clear()
    # # for i in range(1, 100):
    # #     arl.lat.read_concepts(i,False)
    # arl.concepts_evaluation()
    #
    # # arl.rules_scatter(concepts_df[concepts_df.lift > 1])
    # print(len(arl.lat.concepts))
    # print(len(binary[binary[param] == 1]))
    #
    # count_df = pd.DataFrame(index = binary.index, columns=['Prediction', 'Fact'])
    # for i in binary.index:
    #     count_df.loc[i, 'Prediction'] = arl.get_prediction(get_row(binary, i))
    #     count_df.loc[i, 'Fact'] = binary.iloc[i, -1]


    """
for i in set.difference(arl.lat.concepts[3]['B'], {'DAY_BEFORE_KONUS'}):
...     print(range_df[range_df.interval == i].index.values[0])
...     print(range_df[range_df.interval == i]['min'].values[0], ' - ', range_df[range_df.interval == i]['max'].values[0])
    
    confidence_df = concepts_df[(concepts_df.lift > 1)&(concepts_df.confidence > 0.6)&(concepts_df.confidence < 0.8)]
    
    for i in confidence_df.index:
...     print(i, ': ', set.difference(arl.lat.concepts[i]['B'], {'DAY_BEFORE_KONUS'}),'--> {\'DAY_BEFORE_KONUS\'}')
...     for j in arl.lat.concepts[i]['A']:
...         print('VANNA: ', arl.df.loc[j,'VANNA'], ', DATE: ', arl.df.loc[j,'RDATE'])
    """
