from collections.abc import Set  # Для работы с множествами (устаревшее, в новых версиях Python используйте collections.abc)
import numpy  # Для математических операций
# import unidecode  # Для работы с Unicode (не используется в текущем коде)
import numpy as np  # Для математических операций (альтернативное имя)
import pandas as pd  # Для работы с табличными данными
import time  # Для измерения времени выполнения
# import networkx as nx  # Для работы с графами (не используется в текущем коде)
import matplotlib.pyplot as plt  # Для визуализации данных
import joblib  # Для сохранения/загрузки моделей
# from networkx.drawing.nx_agraph import graphviz_layout  # Для визуализации графов (не используется)
from fca_lab import fca_lattice  # Для работы с решетками формальных понятий (FCA)
import arl_binarization  # Модуль для бинаризации данных
import pickle  # Для сериализации объектов
import arl_data  # Специфичный модуль для работы с данными

class arl_fca_lab:
    """
    Класс для анализа данных электролизеров с использованием Formal Concept Analysis (FCA).
    Позволяет:
    - Анализировать взаимосвязи параметров
    - Оценивать правила и их достоверность
    - Визуализировать результаты
    - Прогнозировать дефекты
    """
    
    def __init__(self, bin_type: arl_binarization.BinarizationType = arl_binarization.BinarizationType.HISTOGRAMS,
                 ind_type: bool = True, keep_nan=True, days_before_defect=3):
        """
        Инициализация объекта анализатора.
        
        Параметры:
        ----------
        bin_type : arl_binarization.BinarizationType, default=HISTOGRAMS
            Тип бинаризации данных (STDDEV, QUARTILES, HISTOGRAMS)
        ind_type : bool, default=True
            Индивидуальная модель для каждого электролизера (True) или общая модель (False)
        keep_nan : bool, default=True
            Сохранять ли столбцы для NaN значений в бинарной матрице
        days_before_defect : int, default=1
            Количество дней до дефекта для анализа
        """
        self.bin_type = bin_type  # Метод бинаризации
        self.ind_type = ind_type  # Тип модели (индивидуальная/общая)
        self.keep_nan = keep_nan  # Флаг сохранения NaN значений
        self.days_before_defect = days_before_defect  # Дней до дефекта для анализа
        self.lat = None  # Решетка формальных понятий (будет инициализирована позже)
        self.concepts_df = None  # DataFrame с концептами (устаревшее, используется confidence_df)
        self.confidence_df = {}  # Словарь с оценками правил для каждого объекта
        self.defect = []  # Список целевых параметров (дефектов)
        self.objs = []  # Список объектов (электролизеров) для анализа
        self.bin_matrix = arl_binarization.ArlBinaryMatrix()  # Объект для бинаризации данных
        self.anomalies_only = False  # Флаг работы только с аномальными значениями
        self.threshold = 0.34  # Порог уверенности для правил

    def fit_model(self, df: pd.DataFrame, defect: str, obj_column: str, parse_date: str, anomalies_only: bool=False):
        """
        Построение модели анализа данных.
        
        Параметры:
        ----------
        df : pd.DataFrame
            Исходные данные для анализа
        defect : str
            Название столбца с целевым параметром (дефектом)
        obj_column : str
            Название столбца с номерами электролизеров
        parse_date : str
            Название столбца с датами измерений
        anomalies_only : bool, default=False
            Использовать только аномальные значения для анализа

        Результаты сохраняются в атрибутах класса
        """
        self.defect = defect  # Сохраняем название целевого параметра
        # Определяем список объектов для анализа
        if self.ind_type:
            self.objs = list(df.index.get_level_values(0).unique())  # Все уникальные электролизеры
        else:
            self.objs = ['all']  # Общая модель для всех данных
            
        self.anomalies_only = anomalies_only  # Сохраняем флаг аномалий

        # Создаем модель бинаризации
        self.bin_matrix.create_model(df, obj_column, parse_date, self.bin_type, defect, self.ind_type)
        print('Модель создана')
        
        # Преобразуем данные в бинарный формат
        self.bin_matrix.transform(df, self.defect, obj_column, parse_date, self.keep_nan,
                                self.days_before_defect, self.anomalies_only)
        print('Данные бинаризованы')

        # Для каждого объекта (электролизера) строим решетку концептов
        for obj in self.objs:
            start_time = time.time()  # Засекаем время
            
            # Подготовка данных для анализа:
            # Берем только строки за 1 день до дефекта
            if not(obj == 'all'):
                # Для индивидуальной модели фильтруем по конкретному электролизеру
                pdf = self.bin_matrix.binary[
                    (self.bin_matrix.binary['defect_markup'] == 1) & 
                    (self.bin_matrix.binary[obj_column] == obj)].drop(
                    [obj_column, parse_date], axis='columns')
            else:
                # Для общей модели берем все данные
                pdf = self.bin_matrix.binary[(self.bin_matrix.binary['defect_markup'] == 1)].drop(
                    [obj_column, parse_date], axis='columns')
            
            # Инициализация решетки формальных понятий
            lat = fca_lattice(pdf)
            print("Объект ", obj, "\nКол-во нарушений:", len(pdf))
            
            # Генерация концептов
            lat.in_close(0, 0, 0)
            print(":\nгенерация концептов --- %s seconds ---" % (time.time() - start_time))

            # Оценка концептов (правил)
            start_time = time.time()
            self.confidence_df[obj] = self.rules_evaluation(lat, self.bin_matrix.binary, pdf)
            print("оценка концептов --- %s seconds ---" % (time.time() - start_time))

    def rules_evaluation(self, lat: fca_lattice, df: pd.DataFrame, pdf: pd.DataFrame):
        """
        Оценка правил (концептов) по показателям уверенности и поддержки.
        
        Параметры:
        ----------
        lat : fca_lattice
            Решетка формальных понятий
        df : pd.DataFrame
            Полная бинарная матрица
        pdf : pd.DataFrame
            Матрица с данными за 1 день до дефекта
            
        Возвращает:
        -----------
        pd.DataFrame
            Таблица с оценками правил (уверенность, поддержка, мощность)
        """
        # Инициализация DataFrame для хранения оценок
        confidence_df = pd.DataFrame(
            index=range(len(lat.concepts)), 
            columns=['B', 'confidence', 'd_support']
        )
        pdf_len = len(pdf)  # Количество примеров дефектов
        
        # Подготовка данных для вычисления производных
        df_derivation = pd.Series(index=lat.context.columns, dtype='object')
        for col in df_derivation.index:
            # Для каждого столбца сохраняем индексы строк, где значение = 1
            df_derivation.loc[col] = set(df.loc[:, col][df.loc[:, col] == 1].index)

        # Оценка каждого концепта
        for i in range(len(lat.concepts)):
            # Пропускаем пустые концепты
            if (lat.concepts[i]['B'].difference({'defect_markup'}) != set()) and (lat.concepts[i]['A'] != set()):
                # Вычисляем мощность левой части правила (без целевого параметра)
                left_extent = len(
                    set.intersection(*[df_derivation[j] for j in lat.concepts[i]['B'].difference({'defect_markup'})])
                )
                
                # Вычисляем мощность всего правила (с целевым параметром)
                rule_extent = len(lat.concepts[i]['A'])
                
                # Вычисляем уверенность правила (confidence)
                confidence_df.loc[i, 'confidence'] = rule_extent / left_extent
                
                # Вычисляем поддержку правила (support)
                confidence_df.loc[i, 'd_support'] = rule_extent / pdf_len
                
                # Сохраняем атрибуты концепта
                confidence_df.loc[i, 'B'] = lat.concepts[i]['B']

        # Сортируем правила по уверенности (по убыванию)
        confidence_df = confidence_df.sort_values(by='confidence', axis=0, ascending=False)
        return confidence_df

    def rules_scatter(self, obj_num, how: ['show', 'save'] = 'show', alpha=0.8, s=50):
        """
        Визуализация правил на диаграмме рассеяния (уверенность vs поддержка).
        
        Параметры:
        ----------
        obj_num : str/int
            Номер объекта (электролизера) или 'all' для общей модели
        how : str, default='show'
            'show' - отобразить график, 'save' - сохранить в файл
        alpha : float, default=0.8
            Прозрачность точек
        s : int, default=50
            Размер точек
            
        """
        fig, ax = plt.subplots()
        weight_df = self.confidence_df[obj_num].dropna()  # Удаляем пустые значения
        
        # Создаем диаграмму рассеяния:
        # X - уверенность, Y - поддержка, цвет - мощность правила
        scatter = ax.scatter(
            x=weight_df['confidence'], 
            y=weight_df['d_support'],
            c=[len(weight_df.loc[i, 'B']) for i in weight_df.index],
            cmap='viridis', 
            alpha=alpha, 
            s=s, 
            label=weight_df['confidence']
        )
        
        # Добавляем легенду для цвета (мощность правила)
        legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Мощность правила")
        ax.add_artist(legend1)
        
        # Настройка осей и заголовков
        plt.xlabel("Уверенность")
        plt.ylabel("Поддержка")
        
        # Отображение или сохранение графика
        if how == 'show':
            plt.show()
        else:
            plt.savefig('scatter.png')

    def get_prediction(self, row_set: set, obj_num, support_threshold=0):
        """
        Прогнозирование дефекта для набора атрибутов.
        
        Параметры:
        ----------
        row_set : set
            Множество атрибутов для анализа
        obj_num : str/int
            Номер объекта (электролизера) или 'all' для общей модели
        support_threshold : int, default=0
            Порог поддержки (не используется в текущей реализации)
            
        Возвращает:
        -----------
        tuple (confidence, row_index) или None
            confidence - уверенность срабатывания правила
            row_index - индекс правила в confidence_df
            None - если подходящее правило не найдено
        """
        # Ищем первое правило, которое соответствует входным данным
        for row_index in self.confidence_df[obj_num].index:
            if self.confidence_df[obj_num].loc[row_index, 'confidence'] >= self.threshold:
                rule_set = self.confidence_df[obj_num].loc[row_index, "B"]
                if isinstance(rule_set, Set):
                    # Проверяем, что все атрибуты правила (кроме целевого) есть в строке
                    if rule_set.difference(row_set) == {'defect_markup'}:
                        return self.confidence_df[obj_num].loc[row_index, 'confidence'], rule_set
        return None

def get_row(df: pd.DataFrame, num: int = 0):
    """
    Вспомогательная функция для извлечения множества атрибутов из строки DataFrame.
    
    Параметры:
    ----------
    df : pd.DataFrame
        Исходный DataFrame
    num : int, default=0
        Номер строки для анализа
        
    Возвращает:
    -----------
    set
        Множество атрибутов, которые равны 1 в указанной строке
    """
    row = df.iloc[num,2:-2]  # Берем строку, исключая первые 2 и последние 2 столбца
    row_set = set()
    for (k, v) in row.items():
        if v:
            row_set.add(k)  # Добавляем атрибуты со значением 1
    return row_set

def dump_model(model, file_name):
    """
    Сохранение модели в файл с использованием pickle.
    
    Параметры:
    ----------
    model : object
        Модель для сохранения
    file_name : str
        Имя файла для сохранения
    """
    picklefile = open(file_name, 'wb')
    pickle.dump(model, picklefile)
    picklefile.close()

def load_model(file_name):
    """
    Загрузка модели из файла.
    
    Параметры:
    ----------
    file_name : str
        Имя файла с сохраненной моделью
        
    Возвращает:
    -----------
    object
        Загруженная модель
    """
    picklefile = open(file_name, 'rb')
    model = pickle.load(picklefile)
    picklefile.close()
    return model


if __name__ == '__main__':
    pass    