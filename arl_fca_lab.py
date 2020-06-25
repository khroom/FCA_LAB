import numpy as np
import pandas as pd
import time
import networkx as nx
import matplotlib.pyplot as plt
import joblib
from networkx.drawing.nx_agraph import graphviz_layout
from fca_lab import fca_lattice


class arl_fca_lab:
    def __init__(self, df: pd.DataFrame, obj_columns : [], goal_param : str = ''):
        """
        Конструктор класса. Инициализирует основные свойства.
        :param df: Полный бинарный датафрейм, по которому будут определятся концепты.
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
        # Инициализация решетки по урзонному контексту
        self.lat = fca_lattice(self.pdf)

    def concepts_support(self):
        """
        Рассчет параметров для концептов относительно исходного контекста.
        :return:
        """
        df_derivation = pd.Series(index=self.lat.context.columns, dtype='object')
        for col in df_derivation.index:
            df_derivation.loc[col] = set(self.df.loc[:, col][self.df.loc[:, col] == 1].index)

        for i in range(len(self.lat.concepts)):
            # Мощность содержания для левой части правила (без целевого параметра)
            self.lat.concepts[i]['left_extent'] = len(
                set.intersection(*[df_derivation[j] for j in self.lat.concepts[i]['B'].difference({self.param})]))
            # Можность содержания для всего правила (с целевым параметром)
            self.lat.concepts[i]['rule_extent'] = len(self.lat.concepts[i]['A'])
            # Мощность объема левой части правила
            self.lat.concepts[i]['rule_intent'] = len(self.lat.concepts[i]['B'])
            # Достоверность правила, как часто срабатывает
            self.lat.concepts[i]['confidence'] = self.lat.concepts[i]['rule_extent'] / self.lat.concepts[i]['left_extent']
            # Поддержка, степень совместности
            self.lat.concepts[i]['lift'] = np.around(
                self.lat.concepts[i]['rule_extent'] * len(binary) ** (self.lat.concepts[i]['rule_intent'] - 1) / \
                np.prod([len(df_derivation.loc[b]) for b in self.lat.concepts[i]['B']]), decimals=1)

    def concepts_evaluation(self, objects: pd.DataFrame, rule_weight_threshold: float):
        """
        Еще одна оценка концептов онтосительно ванн. Попыталась сделать универсальный механизм для оценки концептов
        относительно объектов (ванн в нашем случае)
        :param objects: дополнительный столбцы
        :param rule_weight_threshold: порог для отображения правил. Рассчитываем оценку только для весомых правил.
        :return: Датафрейм оценок
        """
        df = objects
        for i in range(len(self.concepts)):
            rule_weight = self.concepts[i]['rule_extent'] / self.concepts[i]['left_extent']
            if rule_weight > rule_weight_threshold:
                df = pd.concat([df, pd.Series(name=i, dtype="float64")], axis=1)
                for obj_id in set.intersection(*[self.context_derivation_1[j] for j in self.concepts[i]['B']]):
                    df.loc[obj_id, i] = rule_weight
        return df

    def lat_draw(self):
        """
        Рисование решетки
        :return:
        """
        min_w = len(self.concepts[0]['B'])
        pos_list = {}
        # Генерация позиций узлов для отрисовки со случаными координатами по горизонтали
        for i in range(len(self.concepts)):
            pos_list[i] = (np.random.choice(a=[-0.5, 0.5]) * np.random.sample(), min_w - len(self.concepts[i]['B']))
        pos_list[0] = (0, 0)
        # Сглаживание координат узлов алгоритмом Spring
        pos = nx.spring_layout(self.lattice, pos=pos_list, k=10, iterations=3)
        plt.figure(figsize=(25, 25))

        # Отрисовка узлов.
        nx.draw_networkx_nodes(self.lattice, pos, node_color="dodgerblue", node_shape="o")

        # Отрисовка ребер графа
        nx.draw_networkx_edges(self.lattice, pos, edge_color="turquoise", arrows=False, alpha=0.5)
        # nx.draw_networkx_labels(self.lattice, pos)
        # plt.savefig('lattice.png', dpi=300)
        plt.show()

    def lat_draw_threshold(self, rule_weight_threshold: float):
        """
        Рисование решетки. Весомые правила отрисовываются особым образом
        :param rule_weight_threshold: порог весомости правил
        :return:
        """
        pos_list = {}
        min_w = len(self.concepts[0]['B'])

        pos_list = {}
        cnpt_labels = {}
        # Генерация позиций узлов для отрисовки со случаными координатами по горизонтали
        for i in range(len(self.concepts)):
            pos_list[i] = (np.random.choice(a=[-0.5, 0.5]) * np.random.sample(), min_w - len(self.concepts[i]['B']))
        pos_list[0] = (0, 0)
        # Сглаживание координат узлов алгоритмом Spring
        pos = nx.spring_layout(self.lattice, pos=pos_list, k=10, iterations=3)
        plt.figure(figsize=(25, 25))

        # Выбор узлов характерных  для нарушений с порогом веса правила rule_weight_threshold
        hi_nodes = []
        for i in range(0, len(self.concepts)):
            rule_weight = self.concepts[i]['part_support'] / self.concepts[i]['full_support']
            if rule_weight > rule_weight_threshold:
                hi_nodes.append(i)
                cnpt_labels[i] = str(i) + '\nr_w-' + str(np.around(rule_weight / rule_weight_threshold, decimals=2))
        # Список остальных узлов
        lo_nodes = list(set(self.lattice.nodes()).difference(set(hi_nodes)))
        # Отрисовка узлов. Сперва базовых, потом со значимым весом правил
        nx.draw_networkx_nodes(self.lattice, pos, node_color="dodgerblue", node_shape="o", nodelist=lo_nodes)
        nx.draw_networkx_nodes(self.lattice, pos, node_color="salmon", node_shape="^", node_size=150, nodelist=hi_nodes,
                               with_labels=True)
        # Отрисовка ребер графа и подписей для значимых узлов
        nx.draw_networkx_edges(self.lattice, pos, edge_color="turquoise", arrows=False, alpha=0.5)
        nx.draw_networkx_labels(self.lattice, pos, labels=cnpt_labels)
        plt.savefig('lattice.png', dpi=300)
        # plt.show()

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

    # def query_append(self, query: dict, qval: str, axis: [0, 1]):
    #     """
    #     Заготовка функции для эксперимента по поддержке формирования концептов по ИАМ
    #     :param query:
    #     :param qval:
    #     :param axis:
    #     :return:
    #     """
    #     if axis == 1:
    #         for i in self.lattice.successors()
    #     elif axis == 0:


if __name__ == '__main__':
    start_time = time.time()
    binary = pd.read_csv('.\\haz_binary\\10_binary_stddev_true_anomalies_only.csv', index_col=0)
    # binary = pd.read_csv('IAM.csv',index_col=0)
    param = ''
    arl = arl_fca_lab(binary, ['VANNA', 'RDATE'], 'KONUSOV')
    print("Загрузка --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    arl.lat.in_close(0, 0, 0)
    # lat.stack_my_close()
    print("Генерация концептов --- %s seconds ---" % (time.time() - start_time))
    # print(len(arl.lat.concepts))
    # start_time = time.time()
    # lat.fill_lattice()
    # print("Построение решетки--- %s seconds ---" % (time.time() - start_time))
    # start_time = time.time()
    # lat.concepts_support(binary)
    # print("Расчет оценок--- %s seconds ---" % (time.time() - start_time))
