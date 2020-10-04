import numpy as np
import pandas as pd
import time
import networkx as nx
import matplotlib.pyplot as plt
import joblib
from networkx.drawing.nx_agraph import graphviz_layout
from fca_lab import fca_lattice


class arl_fca_lab:
    def __init__(self, df: pd.DataFrame, obj_columns: [], goal_param: str = ''):
        """
        Конструктор класса. Инициализирует основные свойства.
        :param df: Полный бинарный датафрейм, по которому будут определяться концепты.
        :param param: Целевой параметр из числа столбцов df. По умолчанию пустая строка.
        :param obj_columns: Список названий столбцов, относящихся к описанию объекта наблюдения
        """
        self.df = df
        self.param = goal_param
        self.obj_columns = obj_columns
        """
        Подготовка контекста для расчета решетки - pdf. Из исходного датасета описательные столбцы,
        которые не содержат числовых параметров и относятся к описанию объектов исследования. Оставляем только
        строки у которых в столбце целевого параметра стоит 1.
        """
        self.pdf = df[df[goal_param] == 1].drop(obj_columns, axis='columns')
        # Инициализация решетки по урезанному контексту
        self.lat = fca_lattice(self.pdf)

    def concepts_evaluation(self):
        """
        Рассчет параметров для концептов относительно исходного контекста.
        :return:
        """
        df_derivation = pd.Series(index=self.lat.context.columns, dtype='object')
        for col in df_derivation.index:
            df_derivation.loc[col] = set(self.df.loc[:, col][self.df.loc[:, col] == 1].index)
        param_fraction = len(self.df[self.df[self.param] == 1])/len(self.df)
        concepts_df = pd.DataFrame(index=range(len(self.lat.concepts)),
                                   columns=['param_support', 'support', 'confidence', 'lift', 'cardinality', 'intent'])
        df_len = len(self.df)
        pdf_len = len(self.pdf)

        for i in range(len(self.lat.concepts)):
            # Мощность содержания для левой части правила (без целевого параметра)
            self.lat.concepts[i]['left_extent'] = len(
                set.intersection(*[df_derivation[j] for j in self.lat.concepts[i]['B'].difference({self.param})]))
            # Можность содержания для всего правила (с целевым параметром)
            self.lat.concepts[i]['rule_extent'] = len(self.lat.concepts[i]['A'])
            # Поддержка по нарушениям
            concepts_df.loc[i, 'param_support'] = self.lat.concepts[i]['rule_extent'] / pdf_len
            # Поддержка правила
            concepts_df.loc[i, 'support'] = self.lat.concepts[i]['rule_extent']/df_len
            # Мощность объема левой части правила
            concepts_df.loc[i, 'cardinality'] = len(self.lat.concepts[i]['B'])
            # Достоверность правила, как часто срабатывает
            concepts_df.loc[i,'confidence'] = self.lat.concepts[i]['rule_extent']/self.lat.concepts[i]['left_extent']
            # Поддержка, вклад левой части в вероятность правой части
            concepts_df.loc[i, 'lift'] = concepts_df.loc[i,'confidence']/param_fraction
            # Количество нарушений покрытых правилом
            concepts_df.loc[i, 'intent'] = len(self.lat.concepts[i]['A'])

        return concepts_df

    """
    def concepts_evaluation(self, objects: pd.DataFrame, rule_weight_threshold: float):
    """
        # Еще одна оценка концептов онтосительно ванн. Попыталась сделать универсальный механизм для оценки концептов
        # относительно объектов (ванн в нашем случае)
        # :param objects: дополнительный столбцы
        # :param rule_weight_threshold: порог для отображения правил. Рассчитываем оценку только для весомых правил.
        # :return: Датафрейм оценок
    """
        df = objects
        for i in range(len(self.concepts)):
            rule_weight = self.concepts[i]['rule_extent'] / self.concepts[i]['left_extent']
            if rule_weight > rule_weight_threshold:
                df = pd.concat([df, pd.Series(name=i, dtype="float64")], axis=1)
                for obj_id in set.intersection(*[self.context_derivation_1[j] for j in self.concepts[i]['B']]):
                    df.loc[obj_id, i] = rule_weight
        return df
    """


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

    def rules_scatter(self, weight_df: pd.DataFrame, alpha=1, s=10):
        """
        Диаграмма рассеяния для весомых правил
        :param weight_df: Датафрейм оценок в разрезе ванн
        :param alpha: прозрачность
        :param s: размер маркеров
        :return:
        """
        sc_df = weight_df.iloc[:, 2:]
        sc_df.index = range(len(weight_df.index))
        sc_df.columns = range(len(sc_df.columns))
        df_stack = sc_df.stack()
        df_stack = df_stack.reset_index()
        df_stack.columns = [self.param, 'RULE', 'WEIGHT']
        plt.clf()
        plt.scatter(x=list(df_stack[self.param]), y=list(df_stack['RULE']), c=list(df_stack['WEIGHT']), cmap='viridis',
                    alpha=1, s=10, label=list(df_stack['WEIGHT']))
        # plt.legend()
        plt.savefig('scatter.png')


if __name__ == '__main__':
    start_time = time.time()
    binary = pd.read_csv('.\\haz_binary\\10_binary_stddev_true_anomalies_only.csv', index_col=0, sep=',')
    # binary = pd.read_csv('IAM.csv',index_col=0)
    param = ''
    binary.drop(['KONUSOV'], axis='columns')
    arl = arl_fca_lab(binary, ['VANNA', 'RDATE'], 'DAY_BEFORE_KONUS')
    print("Загрузка --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    # arl.lat.in_close(0, 0, 0)
    arl.lat.stack_my_close(100)
    print("Генерация концептов --- %s seconds ---" % (time.time() - start_time))
    arl.lat.concepts.clear()
    for i in range(1, 100):
        arl.lat.read_concepts(i,False)
    concepts_df = arl.concepts_evaluation()
    confidence_df = concepts_df[concepts_df.lift > 1]
    fig, ax = plt.subplots()
    # fig = plt.figure(figsize=(9, 9))
    scatter = ax.scatter(x=confidence_df['confidence'], y=confidence_df['intent'], c=confidence_df['cardinality'],
                         cmap='viridis', alpha=.6, s=50, label=confidence_df['cardinality'])
    legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="cardinality")
    ax.add_artist(legend1)
    plt.xlabel("Confidence")
    plt.ylabel("Intent")
    # plt.show()
    plt.savefig('scatter.png')

    """
    confidence_df = concepts_df[(concepts_df.lift > 1)&(concepts_df.confidence > 0.6)&(concepts_df.confidence < 0.8)]
    
    for i in confidence_df.index:
...     print(i, ': ', set.difference(arl.lat.concepts[i]['B'], {'DAY_BEFORE_KONUS'}),'--> {\'DAY_BEFORE_KONUS\'}')
...     for j in arl.lat.concepts[i]['A']:
...         print('VANNA: ', arl.df.loc[j,'VANNA'], ', DATE: ', arl.df.loc[j,'RDATE'])
    """