import numpy as np
import pandas as pd
import time
import networkx as nx
import matplotlib.pyplot as plt
import joblib
from networkx.drawing.nx_agraph import graphviz_layout


class fca_lattice:
    def __init__(self, df: pd.DataFrame):
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
        # определяем супремум и инфимум решетки
        self.concepts = [{'A': set(self.context.index), 'B': set()}, {'A': set(), 'B': set(self.context.columns)}]
        # проверить следующую строку
        self.threshold_base = len(self.context.index)

        # Множество концептов для быстрого расчета. Генерится только объем и хранится в виде кортежа (хешируемый тип)
        self.concepts_set = set()

        self.columns_len = len(self.context.columns)
        self.index_len = len(self.context.index)
        self.step_count = 0
        self.stack_intervals = pd.DataFrame()
        self.stack = []

        # Предварительный расчет обемов для каждого столбца и содержаний для каждой строки. Ускоряет расчет концептов.
        self.context_derivation_0 = pd.Series(index=self.context.index, dtype='object')
        self.context_derivation_1 = pd.Series(index=self.context.columns, dtype='object')
        for i in range(0, self.index_len):
            self.context_derivation_0.iloc[i] = self.derivation(self.context.index[i], 0)
        for i in range(0, self.columns_len):
            self.context_derivation_1.iloc[i] = self.derivation(self.context.columns[i], 1)
        # Инициализация двунаправленного графа для представления решетки
        self.lattice = nx.DiGraph()
        # Инициализация двунаправленного графа для экспериментальной решетки с маркированными ребрами (пока не вышло) для ускорения
        # выполнения запросов к ИАМ. Надо бы разделить ИАМ от простого АФП и от Ассоциативных правил.
        # self.lbl_lattice = nx.DiGraph()

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
        for j in range(column, self.columns_len):
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

    def __my_close__(self, column: int, concept_A: set, step_n: int):
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
            tp_concept_a = tuple(sorted(new_concept_a))
            if (new_concept_a_len > self.stack_intervals.loc[step_n, 'left']) & (
                    new_concept_a_len <= self.stack_intervals.loc[step_n, 'right']):
                if tp_concept_a not in self.concepts_set:
                    self.concepts_set.add(tp_concept_a)
                    print('\r', len(self.concepts_set), end='')
                    self.__my_close__(j + 1, new_concept_a, step_n)
            elif (new_concept_a_len < self.stack_intervals.loc[step_n, 'left']) & (new_concept_a_len > 0):
                # print('\r', new_concept_a_len, end='')
                ind = self.stack_intervals[(self.stack_intervals['left'] < new_concept_a_len) & (self.stack_intervals['right'] >= new_concept_a_len)].index.values[0]
                # добавление параметров в стек вызова
                if (tp_concept_a not in self.stack[ind]) or (self.stack[ind][tp_concept_a] > j+1):
                    self.stack[ind].update({tp_concept_a: j+1})

    def stack_my_close(self, step_count: int = 100):
        """
        Управление стеком параметров вызова функции my_close
        :param step_count: количесвто шагов расчета
        :return:
        """
        # Шаг расчета
        self.step_count = step_count
        step = self.index_len / step_count
        # Интервалы для быстрого расчета концептов. Левая и правая границы.
        self.stack_intervals = self.stack_intervals.reindex(index=range(step_count))
        self.stack_intervals['left'] = [np.around(step * (step_count - i)) for i in range(1, step_count + 1)]
        self.stack_intervals['right'] = [np.around(step * (step_count - i)) for i in range(step_count)]
        # Стек параметров вызова функции для каждого интервала. Позваляет постепенно опускаться вглубь,
        # расчитывая сперва самые большие по объему концепты.
        self.stack = [{} for i in range(step_count)]

        concept_count = 0
        # инициализация первого интервала супремумом
        self.stack[0].update({tuple(sorted(set(self.context.index))): 0})
        # проход по интервалам
        for i in range(step_count):
            # печать информации о списке параметров вызова в интервале
            print(i,', interval: ', self.stack_intervals.loc[i, 'left'], ' - ', self.stack_intervals.loc[i, 'right'],
                  ', stack: ', len(self.stack[i]))
            # вызов функци определения концептов с сохраненными параметрвами вызова
            for k in self.stack[i].keys():
                self.__my_close__(self.stack[i][k], set(k), i)
            # подсчет общего числа концептов
            concept_count = concept_count + len(self.concepts_set)
            print('\n', 'concepts: ', len(self.concepts_set), '/', concept_count)
            # выгрузка найденных концептов в файл, очистка списка концептов и стека вызова для интервала
            joblib.dump(self.concepts_set, ".\\result\\concepts_set" + str(i) + ".joblib")
            self.concepts_set.clear()

    def stack_concepts_repair(self, ):
        """
        Загрузка концептов расчитанных пошагово. Надо подумть как лучше сделать ,если количество шагов расчета
        не является свойстом решетки, а задается параметром
        :return:
        """

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
        Заполняет двунаправленный граф (решетку). Пересмотреть расчет ребер с инфимумом и генерацию лейблов ребер!!!
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
                                             add_d=','.join(str(s) for s in self.concepts[i]['B'] - self.concepts[j]['B']),
                                             add_m=','.join(str(s) for s in self.concepts[j]['A'] - self.concepts[i]['A']))


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

    def lattice_query_support(self, axis, el, bound_n):
        if axis == 'A':
            if el in lat.concepts[bound_n]['A']:
                return bound_n
            else:
                items_list = list(lat.lattice.pred[bound_n].items())
                for n in range(len(list(self.lattice.pred[bound_n].items()))):
                    if el in list(self.lattice.pred[bound_n].items())[n][1]['add_m'].split(','):
                        return list(self.lattice.pred[bound_n].items())[n][0]
        elif axis == 'B':
            if el in lat.concepts[bound_n]['B']:
                return bound_n
            else:
                items_list = list(lat.lattice.succ[bound_n].items())
                for n in range(len(list(self.lattice.succ[bound_n].items()))):
                    if el in list(self.lattice.succ[bound_n].items())[n][1]['add_d'].split(','):
                        return list(self.lattice.succ[bound_n].items())[n][0]
        else:
            return 0

if __name__ == '__main__':
    binary = pd.read_csv('IAM_random.csv', index_col=0)
#   Инициализация объекта
    lat = fca_lattice(binary)
    print("Загрузка --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
#     Вызов процедуры расчета решетки. in_close - классический расчет для небольших контекстов, 
#     stack_my_close - пошаговый расчет (считает только одну часть концептов)
    # lat.in_close(0, 0, 0)
    lat.stack_my_close(20)
    # lat.my_close(0, set(lat.context.index))
    print("Генерация концептов --- %s seconds ---" % (time.time() - start_time))
    # print(len(lat.concepts))
    start_time = time.time()
#     построение решетки еще в работе обнаружена ошибка
    lat.fill_lattice()
    print("Построение решетки--- %s seconds ---" % (time.time() - start_time))
