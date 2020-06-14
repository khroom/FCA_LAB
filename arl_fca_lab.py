import numpy as np
import pandas as pd
import time
import networkx as nx
import matplotlib.pyplot as plt
import joblib
from networkx.drawing.nx_agraph import graphviz_layout
from fca_lab import fca_lattice


class ARL_fca_lab:
    def __init__(self, df: pd.DataFrame, param:str = '', step_count: int = 100):
        """
        Конструктор класса. Инициализирует основные свойства.
        :param df: Полный бинарный датафрейм, по которому будут определятся концепты.
        :param param: Целевой параметр из числа столбцов df. По умолчанию пустая строка.
        :param step_count: Количество шагов для расчета концептов большого контекста. По умолчанию 100.
        Возможно по умолчанию лучше 0, чтобы иметь возможность простого рекурсивного расчета.
        TODO
        В идеале хотелось бы загружать исходную таблицу и накладывать фильтр по выбранному целевому параметру,
        для того чтобы вычислять концепты по сокращенной выборке, а оценки считать по полной.
        """
        self.context = df
        self.param = param

        # возможно уже не нужны. Явно нужно пересмотреть
        if param:
            self.concepts = [{'A': self.derivation(param, 1), 'B': {param}}]
            # self.context.drop([param], axis='columns')
            # проверить следующую строку
            self.threshold_base = len(self.concepts[0]['A'])
        else:
            self.concepts = [{'A': set(self.context.index), 'B': set()}, {'A': set(), 'B': set(self.context.columns)}]
            #
            # проверить следующую строку
            self.threshold_base = len(self.context.index)

        # Множество концептов для быстрого расчета. Генерится только объем и хранится в виде кортежа (хешируемый тип)
        self.concepts_set = set()
        self.columns_len = len(self.context.columns)
        self.index_len = len(self.context.index)
        self.step_count = step_count
        # Шаг расчета
        step = self.index_len / step_count
        # Интервалы для быстрого расчета концептов. Левая и правая границы.
        self.stack_intervals = pd.DataFrame(index=range(step_count))
        self.stack_intervals['left'] = [np.around(step * (step_count - i)) for i in range(1, step_count + 1)]
        self.stack_intervals['right'] = [np.around(step * (step_count - i)) for i in range(step_count)]
        # Стек параметров вызова функции для каждого интервала. Позваляет постепенно опускаться вглубь,
        # расчитывая сперва самые большие по объему концепты.
        self.stack = [[] for i in range(self.step_count)]
        # Предварительный расчет обемов для каждого столбца и содержаний для каждой строки. Ускоряет расчет концептов.
        self.context_derivation_0 = pd.Series(index=self.context.index, dtype='object')
        self.context_derivation_1 = pd.Series(index=self.context.columns, dtype='object')
        for i in range(0, len(self.context.index)):
            self.context_derivation_0.iloc[i] = self.derivation(self.context.index[i], 0)
        for i in range(0, len(self.context.columns)):
            self.context_derivation_1.iloc[i] = self.derivation(self.context.columns[i], 1)
        # Инициализация двунаправленного графа для представления решетки
        self.lattice = nx.DiGraph()
        # Инициализация двунаправленного графа для экспериментальной решетки с маркированными ребрами (пока не вышло) для ускорения
        # выполнения запросов к ИАМ. Надо бы разделить ИАМ от простого АФП и от Ассоциативных правил.
        self.lbl_lattice = nx.DiGraph()

    def is_cannonical(self, column, new_a, r):
        """
        Проверка концепта на каноничность. Классический алгоритм
        :param column: номер столца на основе которого сгенерирован концепт
        :param new_a: объем нового концепта, который нужно проверить на каноничность
        :param r: номер концепта на основе которго сгененирован новый концепт
        :return: результат проверки
        """
        for i in range(column, -1, -1):
            if self.context.columns[i] not in self.concepts[r]['B']:
                if new_a.issubset(self.context_derivation_1.iloc[i]):
                    return False
        return True

    def in_close(self, column: int, r: int, threshold=0.0):
        """
        Закрывает концепт по контексту. Классический алгоритм
        :param column: номер столбца с которого начать проверку
        :param r: номер текущего контекста
        :param threshold: порог (использовался при расчете концептов заданного объема)
        :return:
        """
        for j in range(column, len(self.context.columns)):
            new_concept = {'A': self.context_derivation_1.iloc[j].intersection(self.concepts[r]['A']), 'B': set()}
            if len(new_concept['A']) == len(self.concepts[r]['A']):
                self.concepts[r]['B'].add(self.context.columns[j])
            else:
                if (len(new_concept['A']) != 0) and (len(new_concept['A']) > self.threshold_base * threshold):
                    if self.is_cannonical(j - 1, new_concept['A'], r):
                        new_concept['B'] = new_concept['B'].union(self.concepts[r]['B'])
                        new_concept['B'].add(self.context.columns[j])
                        self.concepts.append(new_concept)
                        self.in_close(j + 1, len(self.concepts) - 1, threshold)

    def my_close(self, column: int, concept_A: set, step_n: int):
        """
        Оригинальный алгоритм поиска концептов по шагам
        :param column: номер столбца
        :param concept_A: объем концепта
        :param step_n: шаг расчета
        :return:
        """
        tp_concept_a = tuple(sorted(concept_A))
        if tp_concept_a not in self.concepts_set:
            self.concepts_set.add(tp_concept_a)

        for j in range(column, self.columns_len):
            new_concept_a = concept_A.intersection(self.context_derivation_1.iloc[j])
            new_concept_a_len = len(new_concept_a)
            if (new_concept_a_len > self.stack_intervals.loc[step_n, 'left']) & (
                    new_concept_a_len <= self.stack_intervals.loc[step_n, 'right']):
                tp_concept_a = tuple(sorted(new_concept_a))
                if tp_concept_a not in self.concepts_set:
                    self.concepts_set.add(tp_concept_a)
                    print('\r', len(self.concepts_set), end='')
                    self.my_close(j + 1, new_concept_a, step_n)
            elif (new_concept_a_len < self.stack_intervals.loc[step_n, 'left']) & (new_concept_a_len > 0):
                # print('\r', new_concept_a_len, end='')
                ind = self.stack_intervals[(self.stack_intervals['left'] < new_concept_a_len) & (self.stack_intervals['right'] >= new_concept_a_len)].index.values[0]
                self.stack[ind].append((j+1, new_concept_a, ind))

    def stack_my_close(self):
        """
        Управление стеком параметров вызова функции my_close
        :return:
        """
        concept_count = 0
        self.stack[0].append((0, set(self.context.index), 0))
        for i in range(self.step_count):
            print(i,', interval: ', self.stack_intervals.loc[i, 'left'], ' - ', self.stack_intervals.loc[i, 'right'],
                  ', stack: ', len(self.stack[i]))
            while self.stack[i]:
                self.my_close(*self.stack[i].pop())

            concept_count = concept_count + len(self.concepts_set)
            print('\n', 'concepts: ', len(self.concepts_set), '/', concept_count)
            joblib.dump(self.concepts_set, ".\\result\\concepts_set" + str(i) + ".joblib")
            self.concepts_set.clear()
            self.stack[i].clear()

    def derivation(self, q_val: str, axis=0):
        """
        Вычисляет по контексту множество штрих для одного элемента (строка или столбец)
        :param q_val: индекс столбца или строки
        :param axis: ось (1 - стобец, 0 - строка)
        :return: результат деривации (операции штрих)
        """
        if axis == 1:
            # поиск по измерениям (столбцам)
            tmp_df = self.context.loc[:, q_val]
        else:
            # поиск по показателям (строкам)
            tmp_df = self.context.loc[q_val, :]
        return set(tmp_df[tmp_df == 1].index)

    def fill_lattice(self):
        """
        Заполняет двунаправленный граф (решетку)
        :return:
        """
        # сортируем множество концептов по мощности объема. Навводила разные ключи в словарь, надо бы упорядочить.
        for i in range(len(self.concepts)):
            self.concepts[i]['W'] = len(self.concepts[i]['A'])
        self.concepts = sorted(self.concepts, key=lambda concept: concept['W'], reverse=True)

        for i in range(len(self.concepts)):
            self.lattice.add_node(i, ext_w=self.concepts[i]['W'],
                                  intent=','.join(str(s) for s in self.concepts[i]['B']))
            for j in range(i - 1, -1, -1):
                if (self.concepts[j]['B'].issubset(self.concepts[i]['B'])) & (
                        self.concepts[i]['A'].issubset(self.concepts[j]['A'])):
                    if not nx.has_path(self.lattice, j, i):
                        self.lattice.add_edge(j, i,
                                              add_col=','.join(
                                                  str(s) for s in self.concepts[j]['B'] - self.concepts[i]['B']))

    def concepts_support(self, binary: pd.DataFrame):
        """
        Рассчитываю параметры для концептов относительно исходного контекста.
        :param binary: исходный контекст без учета целевого параметра. Может лучше его сразу передавать в класс.
        :return:
        """
        binary_derivation = pd.Series(index=self.context.columns, dtype='object')
        for col in binary_derivation.index:
            binary_derivation.loc[col] = set(binary.loc[:, col][binary.loc[:, col] == 1].index)

        for i in range(len(self.concepts)):
            # Мощность содержания для левой части правила (без целевого параметра)
            self.concepts[i]['left_extent'] = len(
                set.intersection(*[binary_derivation[j] for j in self.concepts[i]['B'].difference({self.param})]))
            # Можность содержания для всего правила (с целевым параметром)
            self.concepts[i]['rule_extent'] = len(self.concepts[i]['A'])
            # Мощность объема левой части правила
            self.concepts[i]['rule_intent'] = len(self.concepts[i]['B'])
            # Достоверность правила, как часто срабатывает
            self.concepts[i]['confidence'] = self.concepts[i]['rule_extent'] / self.concepts[i]['left_extent']
            # Поддержка, степень совместности
            self.concepts[i]['lift'] = np.around(
                self.concepts[i]['rule_extent'] * len(binary) ** (self.concepts[i]['rule_intent'] - 1) / \
                np.prod([len(binary_derivation.loc[b]) for b in self.concepts[i]['B']]), decimals=1)

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
    # binary = pd.read_csv('.\\haz_binary\\10_binary_stddev_true_anomalies_only.csv', index_col=0)
    binary = pd.read_csv('IAM.csv',index_col=0)
    param = ''
    # param_df = binary[binary[param] == 1].drop(['VANNA', 'RDATE'], axis='columns')
    # param_df = param_df.iloc[:, [0, 1, 2, 104]]
    lat = pca_lattice(binary, param)
    print("Загрузка --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    lat.in_close(0, 0, 0)
    # lat.stack_my_close()
    # lat.my_close(0, set(lat.context.index))
    print("Генерация концептов --- %s seconds ---" % (time.time() - start_time))
    print(len(lat.concepts))
    start_time = time.time()
    lat.fill_lattice()
    print("Построение решетки--- %s seconds ---" % (time.time() - start_time))
    # start_time = time.time()
    # lat.concepts_support(binary)
    # print("Расчет оценок--- %s seconds ---" % (time.time() - start_time))
