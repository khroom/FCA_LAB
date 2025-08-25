import numpy as np  # Для математических операций
import pandas as pd  # Для работы с табличными данными
import math  # Математические функции
import datetime  # Работа с датами и временем
from enum import Enum  # Создание перечислений
import joblib  # Для сохранения/загрузки моделей
import arl_data  # Специфичный модуль для работы с данными электролизеров

# Перечисление типов бинаризации
class BinarizationType(Enum):
    """
    Определяет методы бинаризации данных:
    - STDDEV: на основе стандартного отклонения (среднее ± σ)
    - QUARTILES: на основе квартилей (25%, 50%, 75%)
    - HISTOGRAMS: на основе анализа пересечений гистограмм
    """
    STDDEV = 0
    QUARTILES = 1
    HISTOGRAMS = 2

class ArlBinaryMatrix:
    """
    Класс для преобразования параметров электролизеров в бинарную матрицу.
    Позволяет:
    - создавать модели бинаризации
    - сохранять/загружать модели
    - преобразовывать данные в бинарный формат
    - определять границы интервалов
    """

    def create_model(self, df: pd.DataFrame, obj_column: str, parse_date: str, settings,
                     defect=None, days_before_defect=3):
        """
        Создает модель бинаризации параметров электролизеров.
        Метод анализирует исходные данные и определяет границы интервалов для каждого параметра
        в соответствии с выбранным методом бинаризации. Результат сохраняется в атрибуте boundaries.
        
        Параметры:
        ----------
        df : pd.DataFrame
            Исходный DataFrame с данными параметров электролизеров
        obj_column : str
            Название столбца, содержащего номера электролизеров
        parse_date : str
            Название столбца с датами измерений
        bin_type : BinarizationType
            Тип бинаризации (STDDEV, QUARTILES или HISTOGRAMS)
        defect : str или list, optional
            Название целевого параметра/параметров (обязателен для HISTOGRAMS)
        ind_type : bool, default=True
            True - индивидуальная бинаризация для каждого электролизера
            False - общая бинаризация для всех электролизеров
        days_before_defect : int, default=3
            Количество дней до дефекта для разметки тестовых данных
            
        """
        # Сохраняем параметры в атрибуты класса
        self.__obj_column = obj_column  # Приватный атрибут с именем столбца электролизера
        self.__date_column = parse_date # Приватный атрибут с датой измерения
        
        # Проверяем и корректируем индекс DataFrame
        if [obj_column, parse_date] == df.index.names:
            # Если индекс уже правильно установлен
            data_params = df
        else:
            # Если индекс нужно перестроить
            data_params = df.reset_index()
            if (obj_column in data_params.columns) & (parse_date in data_params.columns):
                data_params = data_params.set_index(pd.MultiIndex([obj_column, parse_date], names=[obj_column, parse_date]))
                
        # Обрабатываем параметр defect (может быть строкой или списком)
        if not isinstance(defect, list):
            defect = [defect]
            
        data_params = self.__markup_defects(data_params, df[defect], days_before_defect)
        # Разделяем данные на параметры и события дефектов
        data_events = data_params['defect_markup']
        data_params = df.drop(defect, axis=1)

        pot_ids = data_params.index.get_level_values('POT_ID').unique()
        common_id = pd.Index(['all'])
        #создаем фрейм с двухуровневым пустым индексом в колонках
        self.boundaries = pd.DataFrame(index = pd.Index(pot_ids.append(common_id), name=self.__obj_column), columns=[[],[]])
        
        for parameter in data_params.columns:

            if parameter in settings:
            # Бинаризация методом стандартного отклонения
                if settings[parameter]['binarization'] == BinarizationType.STDDEV:
                    if settings[parameter]['individual']:
                        # Индивидуальная статистика по каждому электролизеру
                        stats = data_params[parameter].groupby(obj_column).agg([np.min, np.mean, np.std, np.max])
                        stats.columns = pd.MultiIndex.from_product([ [parameter],stats.columns.values])
                    else:
                        # Общая статистика для всех электролизеров
                        stats = data_params[parameter].agg([np.min, np.mean, np.std, np.max])
                        stats.index = pd.MultiIndex.from_product([ [parameter],stats.index.values])
                        stats = stats.to_frame().T
                        stats.index = ['all']
                    # Получаем границы интервалов
                    parameter_boundaries = self.__get_boundaries_by_stat(stats)
                    self.boundaries = self.boundaries.join(parameter_boundaries)

                # Бинаризация методом квартилей
                elif settings[parameter]['binarization'] == BinarizationType.QUARTILES:
                    if settings[parameter]['individual']:
                        stats = data_params[parameter].groupby(obj_column).describe()
                        stats.columns = pd.MultiIndex.from_product([ [parameter],stats.columns.values])
                    else:
                        # Общие статистика для всех данных
                        stats = data_params[parameter].describe()
                        stats.index = pd.MultiIndex.from_product([ [parameter],stats.index.values])
                        stats = stats.to_frame().T
                        stats.index = ['all']
                    parameter_boundaries = self.__get_boundaries_by_quartiles(stats)
                    self.boundaries = self.boundaries.join(parameter_boundaries)


                # Бинаризация методом гистограмм
                elif settings[parameter]['binarization'] == BinarizationType.HISTOGRAMS:
                    # if settings[parameter]['individual']:
                        # self.boundaries = pd.DataFrame()
                        # # Обрабатываем каждый электролизер отдельно
                        # for group, data in data_params.groupby(obj_column):
                        #     hist_boundaries = self.__get_boundaries_by_hist(
                        #         data, 
                        #         parameter,
                        #         data_events.loc[data.index], 
                        #         day_before=days_before_defect
                        #     )
                        #     self.boundaries = self.boundaries.append(
                        #         pd.concat([hist_boundaries], keys=[group], names=[obj_column])
                        #     )
                        # self.boundaries.index = self.boundaries.index.droplevel(1)
                    # else:
                        # Обрабатываем все данные вместе
                    parameter_boundaries = self.__get_boundaries_by_hist(
                        data_params,
                        parameter, 
                        data_events, 
                      
                    )
                    if parameter_boundaries is not None:
                        self.boundaries = self.boundaries.join(parameter_boundaries)
            
        # Инициализируем бинарную матрицу
        self.binary = None

    def load_model(self, file: str):
        """
        Загружает модель бинаризации из файла.
        
        Параметры:
        ----------
        file : str
            Путь к файлу с сохраненной моделью (.joblib)
            
        Возвращает:
        -----------
        None
        Загружает атрибуты boundaries, bin_type и ind_type из файла

        """
        # Определяем расширение файла
        if '.' in file:
            model = joblib.load(file)
        else:
            model = joblib.load(file + ".joblib")
            
        # Восстанавливаем атрибуты класса
        self.boundaries = model.boundaries
        self.bin_type = model.bin_type
        self.ind_type = model.ind_type
        self.binary = None

    def save_model(self, file: str):
        """
        Сохраняет модель бинаризации в файл.
        
        Параметры:
        ----------
        file : str
            Путь для сохранения файла
            
        Исключения:
        -----------
        Exception: если модель не была создана (boundaries = None)

        """
        if self.boundaries is not None:
            model = self
            model.binary = None  # Не сохраняем бинарную матрицу
            # Сохраняем с правильным расширением
            if '.' in file:
                joblib.dump(self, file)
            else:
                joblib.dump(self, file + ".joblib")
        else:
            raise Exception("No binary model")

    def transform(self, df: pd.DataFrame, defect, obj_column: str, parse_date: str, settings,
                days_before_defect=3):
        """
        Преобразует данные в бинарную матрицу на основе созданной модели.
        
        Параметры:
        ----------
        df : pd.DataFrame
            Исходные данные для преобразования
        defect : str или list
            Название целевого параметра/параметров
        obj_column : str
            Название столбца с номерами электролизеров
        parse_date : str
            Название столбца с датами
        keep_nan : bool, default=True
            Сохранять ли столбцы для NaN значений
        days_before_defect : int, default=3
            Количество дней до дефекта для добавления соответствующего столбца (разметка данных для тестирования)
        anomalies_only : bool, default=False
            Возвращать только аномальные значения
            
        Возвращает:
        -----------
        pd.DataFrame
            Бинарная матрица, где 1 означает попадание в интервал
            
        Исключения:
        -----------
        Exception: если модель не была создана
        """
        if self.boundaries is None:
            raise Exception("No binary model")
            
        # Проверяем и корректируем индекс DataFrame
        if [obj_column, parse_date] == df.index.names:
            data_params = df
        else:
            data_params = df.reset_index()
            if (obj_column in data_params.columns) & (parse_date in data_params.columns):
                data_params = data_params.set_index(pd.MultiIndex([obj_column, parse_date], names=[obj_column, parse_date]))
                
        # Обрабатываем параметр defect (может быть строкой или списком)
        if not isinstance(defect, list):
            defect = [defect]
            
        # Разделяем данные на параметры и события дефектов
        data_events = df[defect]
        data_params = df.drop(defect, axis=1)
        
        # Выбираем только те параметры, для которых есть границы в модели
        cross_cols = list(set(data_params.columns).intersection(
            self.boundaries.columns.get_level_values(0))
        )
        
        # Объединяем данные с границами интервалов
        full_df = self.__concat_data_boundaries(
            data_params[cross_cols], 
            self.boundaries[cross_cols],
            settings
        )
        odd_columns = full_df.columns
        binary = pd.DataFrame(index = full_df.index)
        # Обработка в зависимости от типа бинаризации
        for parameter in settings:
            if settings[parameter]['binarization'] == BinarizationType.STDDEV:
                self.__classify_by_std(full_df, parameter, settings[parameter]['anomalies_only'], settings[parameter]['keep_nan'])
                # binary = binary.join(classified_data, how='left')

            elif settings[parameter]['binarization'] == BinarizationType.QUARTILES:
                self.__classify_by_quartiles(full_df, parameter, settings[parameter]['anomalies_only'], settings[parameter]['keep_nan'])
                # binary = binary.join(classified_data,how='left')

            elif settings[parameter]['binarization'] == BinarizationType.HISTOGRAMS:
                self.__classify_by_hist(full_df, parameter, settings[parameter]['anomalies_only'], settings[parameter]['keep_nan'])
                # binary = binary.join(classified_data,how='left')
                
        binary = full_df.drop(columns=odd_columns)
        binary = binary.droplevel(level=1, axis=1)
        # Форматируем бинарную матрицу и сохраняем в атрибут
        binary = self.__markup_defects(binary, data_events, days_before_defect)
        self.binary = self.__format_binary(binary)
        
        return self.binary

    def __drop_nan(self, df: pd.DataFrame):
        """
        Внутренний метод для удаления столбцов с NaN значениями.
        
        Параметры:
        ----------
        df : pd.DataFrame
            Бинарная матрица
            
        Возвращает:
        -----------
        pd.DataFrame
            Матрица без столбцов NaN
        """
        nan_cols = [c for c in df.columns if c[-4:] == '_NaN']
        return df.drop(nan_cols, axis='columns')

    def __get_boundaries_by_stat(self, stats: pd.DataFrame):
        """
        Внутренний метод для расчета границ по статистике (среднее ± σ).
        
        Параметры:
        ----------
        stats : pd.DataFrame
            Статистики параметров (min, mean, std, max)
            
        Возвращает:
        -----------
        pd.DataFrame
            Границы интервалов для каждого параметра
        """
        boundaries = pd.DataFrame(index=stats.index)
        for c in stats.columns.get_level_values(0).unique():
            # Границы интервалов:
            # 0 - минимальное значение
            # 1 - mean - std (нижняя граница нормального диапазона)
            # 2 - mean + std (верхняя граница нормального диапазона)
            # 3 - максимальное значение
            boundaries[(c, 1)] = stats[(c, 'mean')] - stats[(c, 'std')]  # Нижняя граница
            boundaries[(c, 2)] = stats[(c, 'mean')] + stats[(c, 'std')]  # Верхняя граница
            
        boundaries.columns = pd.MultiIndex.from_tuples(boundaries.columns)
        return boundaries

    def __get_boundaries_by_quartiles(self, stats: pd.DataFrame):
        """
        Внутренний метод для расчета границ по квартилям.
        
        Параметры:
        ----------
        stats : pd.DataFrame
            Статистики параметров (min, 25%, 50%, 75%, max)
            
        Возвращает:
        -----------
        pd.DataFrame
            Границы интервалов для каждого параметра
        """
        boundaries = pd.DataFrame(index=stats.index)
        for c in stats.columns.get_level_values(0).unique():
            # Границы интервалов по квартилям:
            # 0 - минимальное значение
            # 1 - 25% (Q1)
            # 2 - 50% (медиана)
            # 3 - 75% (Q3)
            # 4 - максимальное значение
            boundaries[(c, 0)] = stats[(c, 'min')]  # Минимальное значение
            boundaries[(c, 1)] = stats[(c, '25%')]  # Первый квартиль
            boundaries[(c, 2)] = stats[(c, '50%')]  # Медиана
            boundaries[(c, 3)] = stats[(c, '75%')]  # Третий квартиль
            boundaries[(c, 4)] = stats[(c, 'max')]  # Максимальное значение
            
        boundaries.columns = pd.MultiIndex.from_tuples(boundaries.columns)
        return boundaries

    def __get_boundaries_by_hist(self, df: pd.Series, parameter, data_events: pd.DataFrame, 
                               interval_width=5, thres=0.1):
        """
        Внутренний метод для расчета границ по пересечению гистограмм.
        
        Параметры:
        ----------
        df : pd.Series
            Данные параметров
        data_events : pd.DataFrame
            Данные о дефектах
        interval_width : int, default=5
            Ширина окна для скользящего среднего
        thres : float, default=0.1
            Порог значимости различий
            
        Возвращает:
        -----------
        pd.DataFrame
            Границы интервалов для каждого параметра
        """
                    
        # Разделяем данные на "чистые" и "с дефектами"
        defect_series = df.loc[data_events[data_events.notna()].index, parameter]
        clean_series = df.loc[df['defect_markup'].isna(), parameter]
        
        boundaries = pd.DataFrame(index=['all'])
        
        # Обрабатываем каждый параметр отдельно
        # for c in df.columns:
            # if df[c].count() == 0:
            #     continue
                
        # Вычисляем оптимальную ширину интервала для гистограммы
        # по формуле Скотта: 3.5 * σ / n^(1/3)
        step = 3.5 * df[parameter].std() / math.pow(df[parameter].count(), 1 / 3)
        bins = []
        clean_hist = []
        defect_hist = []
        cur_value = df[parameter].min()
        
        # Строим гистограммы для "чистых" и "дефектных" данных
        while cur_value < df[parameter].max() + step:
            # Подсчет относительной частоты в текущем интервале
            clean_count = clean_series.loc[
                (clean_series >= cur_value) & 
                (clean_series < cur_value + step)
            ].count() / clean_series.count()
            
            defect_count = defect_series.loc[
                (defect_series >= cur_value) & 
                (defect_series < cur_value + step)
            ].count() / defect_series.count()
            
            # Добавляем только значимые интервалы
            if (clean_count > 0) | (defect_count > 0):
                bins.append(cur_value)
                clean_hist.append(clean_count)
                defect_hist.append(defect_count)
                
            cur_value += step
            
        # Создаем DataFrame с гистограммами
        hist = pd.DataFrame({'clean': clean_hist, 'defect': defect_hist}, index=bins)
        
        # Определяем минимально значимую разницу
        delta = hist.max().max() * thres
        leading = ''  # Текущая доминирующая группа
        i = 0  # Счетчик интервалов
        
        # Анализируем гистограммы для определения границ
        for index, row in hist.iloc[1:].iterrows():
            # Проверяем, есть ли значимые различия в текущем интервале
            if (row['clean'] >= delta) | (row['defect'] >= delta):
                # Вычисляем разницу с использованием скользящего среднего
                change = (
                    (hist['clean'] - hist['defect'])
                    .loc[:index]
                    .tail(interval_width)
                    .rolling(interval_width)
                    .mean()
                ).loc[index]
                
                # Определяем доминирующую группу
                group = 'defect' if change < 0 else 'clean'
                
                # Если доминирующая группа изменилась - фиксируем границу
                if group != leading:
                    if leading == '':
                        # Первая граница
                        leading = group
                    else:
                        # Добавляем границу интервала
                        boundaries[(parameter, (i, leading))] = index
                        leading = group
                        i += 1

        if not boundaries.empty:                
            boundaries.columns = pd.MultiIndex.from_tuples(boundaries.columns)
            return boundaries
        else:
            print(parameter + '   не найдено разделение')

    def decode(self):
        """
        Декодирует модель в таблицу интервалов значений.
        
        Возвращает:
        -----------
        pd.DataFrame
            Таблица с минимальными и максимальными значениями для каждого интервала
            
        Примечание:
        -----------
        Метод используется для интерпретации модели, показывает фактические диапазоны значений
        для каждого параметра и каждого интервала.
        """
        # Фиксированный набор тестовых параметров (пример)
        test = {
            'Vremia pitaniia po bazovoi ustavke_0', 'Vremia APG v ruch.rezh._0', 
            'Kol-vo doz APG v nedopitke_3', 'Vremia otkl. APG_0', 
            'El-lit: temp-ra_3', 'Tsel. napr. el-ra_0', 'Vremia zapreta APG_0', 
            'Doza glinozema_0', 'El-lit: KO_3', 'Cr. vremennaia dobavka_0', 
            'AlF3: ustavka pitaniia_1', 'RMPR: koeff._0', 'Cr. dobavka po shumam_0', 
            'Ustavka APG_0', 'Upravlenie: temp-ra el-ta_3', 'RMPR: vremia zapreta_0', 
            'El-lit: CaF2_0', 'Tsel. temp. el-ta_0', 'Napr. zad._0', 
            'El-lit: MgF2_0', 'Kol-vo doz APG v teste_3', 'Shum_0'
        }
        
        # Подготавливаем структуру для хранения интервалов
        ranges = pd.concat([self.boundaries], keys=['all'], axis=1)
        ranges = ranges.sort_index(axis='index').sort_index(axis='columns')
        
        # Преобразуем границы в интервалы [min, max]
        for c in ranges.columns.get_level_values(1).unique():
            i = 0
            while i < len(ranges[('all', c)].columns) - 1:
                # Для каждой границы создаем пару min-max
                ranges[(c, c + '_' + str(i), 'min')] = ranges[('all', c, i)]
                ranges[(c, c + '_' + str(i), 'max')] = ranges[('all', c, i + 1)]
                i += 1
                
        # Убираем уровень 'all' и преобразуем в удобный формат
        ranges = ranges.drop(['all'], axis=1, level=0)
        ranges = ranges.stack([0, 1])  # Переводим в длинный формат
        ranges = ranges[['min', 'max']]  # Оставляем только min и max
        
        # Дополнительная обработка индексов
        if ranges.index.names[0] == 'all':
            ranges.index = ranges.index.droplevel(0)
            
        # Фильтрация для тестовых параметров (пример)
        t = ranges.loc[6.0].swaplevel(0,1).loc[list(test)]
        t.index = t.index.droplevel(1)
        return t

    def __concat_data_boundaries(self, data: pd.DataFrame, boundaries: pd.DataFrame, settings):
        """
        Внутренний метод для объединения данных с границами интервалов.
        
        Параметры:
        ----------
        data : pd.DataFrame
            Исходные данные параметров
        boundaries : pd.DataFrame
            Границы интервалов из модели
            
        Возвращает:
        -----------
        pd.DataFrame
            Объединенные данные с мультииндексом
        """
        # Подготавливаем данные с мультииндексом столбцов и строк
        df = pd.concat([data], axis='columns', keys=['value'])
        df.columns = df.columns.swaplevel(0, 1)

        df = pd.concat([df], keys=['all'], names=['all'])
        
        for parameter in data.columns:
            if parameter in settings:
                # Объединяем с границами в зависимости от типа модели
                if settings[parameter]['individual']:
                    # Для индивидуальной модели
                    df = df.join(boundaries[[parameter]], on=self.__obj_column, how='inner')
                else:
                    # Для общей модели
                    df = df.join(boundaries[[parameter]], on='all', how='inner')
                    
        df.index = df.index.droplevel(0)
        return df


    def __classify_by_std(self, df: pd.DataFrame, parameter, anomalies_only, keep_nan):
        """
        Внутренний метод классификации по стандартному отклонению.
        
        Параметры:
        ----------
        df : pd.DataFrame
            Данные с границами интервалов
            
        Возвращает:
        -----------
        pd.DataFrame
            Классифицированные данные (номера интервалов)
        """
        # classified = pd.DataFrame(index=df.index)
         
        # Для каждого параметра определяем интервалы
        for c in df[[parameter]].columns.get_level_values(0).unique():
            # Интервал 0: значения ниже (mean - std)
            df.loc[df[(c, 'value')] < df[(c, 1)], c + '_0'] = 1
            # Интервал 2: значения выше (mean + std)
            df.loc[df[(c, 'value')] > df[(c, 2)], c + '_2'] = 1
            
            if not anomalies_only:
                # Интервал 1: значения между (mean - std) и (mean + std)
                df.loc[df[(c, 'value')].between(df[(c, 1)], df[(c, 2)]), c + '_1'] = 1
                    
            if keep_nan:
                df.loc[df[(c, 'value')].isna(), c + '_NaN'] = 1

        

    def __classify_by_quartiles(self, df: pd.DataFrame, parameter, anomalies_only, keep_nan):
        """
        Внутренний метод классификации по квартилям.
        
        Параметры:
        ----------
        df : pd.DataFrame
            Данные с границами интервалов
            
        Возвращает:
        -----------
        pd.DataFrame
            Классифицированные данные (номера интервалов)
        """
        # classified = pd.DataFrame(index=df.index)
        
        # Для каждого параметра определяем интервалы
        for c in df[[parameter]].columns.get_level_values(0).unique():
            # Интервал 0: значения ≤ Q1
            df.loc[df[(c, 'value')] <= df[(c, 1)], c + '_0'] = 1
            # Интервал 3: значения > Q3
            df.loc[df[(c, 'value')] > df[(c, 3)], c + '_3'] = 1
                        
            if not anomalies_only:
                # Интервал 1: значения между Q1 и медианой
                df.loc[
                    (df[(c, 'value')] > df[(c, 1)]) & 
                    (df[(c, 'value')] <= df[(c, 2)]), 
                    c + '_1'
                ] = 1
                # Интервал 2: значения между медианой и Q3
                df.loc[
                    (df[(c, 'value')] > df[(c, 2)]) & 
                    (df[(c, 'value')] <= df[(c, 3)]), 
                    c + '_2'
                ] = 1
                        
            if keep_nan:
                df.loc[df[(c, 'value')].isna(), c + '_NaN'] = 1

        
    def __classify_by_hist(self, df: pd.DataFrame, parameter, anomalies_only, keep_nan):
        """
        Внутренний метод классификации по гистограммам.
        
        Параметры:
        ----------
        df : pd.DataFrame
            Данные с границами интервалов
            
        Возвращает:
        -----------
        pd.DataFrame
            Классифицированные данные (номера интервалов)
        """
        classified = pd.DataFrame(index=df.index)
        
        # Для каждого параметра определяем интервалы
        for c in df[[parameter]].columns.get_level_values(0).unique():
            
            dfc = df[c]
            properties = {}

            for i in dfc.columns:
                if i != 'value':
                    properties[i[0]] = i[1]
                    
            dfc.columns = pd.Index([i if i == 'value' else i[0] for i in dfc.columns])

            for i in dfc.columns:
                if i != 'value':
                    if (not anomalies_only) | (properties[i] == 'defect'):
                        if i+1 in dfc.columns:
                            df.loc[dfc['value'].between(dfc[i], dfc[i+1], inclusive='left'), c + '_' + str(i)] = 1
                        else:
                            df.loc[dfc['value'] > dfc[i], c + '_' + str(i)] = 1


            if not keep_nan:
                df.loc[dfc['value'].isna(), c + '_NaN'] = 1


        # return classified

    def __get_binary(self, df: pd.DataFrame):
        """
        Внутренний метод преобразования в бинарную матрицу.
        
        Параметры:
        ----------
        df : pd.DataFrame
            Классифицированные данные (номера интервалов)
            
        Возвращает:
        -----------
        pd.DataFrame
            Бинарная матрица (one-hot encoding)
        """
        data = df
        binary = pd.DataFrame(index=data.index)
        
        # Для каждого параметра создаем бинарные столбцы
        for c in data.columns:
            i = 0
            # Создаем столбцы для каждого интервала
            while i <= data[c].max():
                binary.loc[data[c] == i, c + '_' + str(int(i))] = 1
                i += 1
                
            # Добавляем столбец для NaN значений
            binary.loc[data[c].isna(), c + '_NaN'] = 1
            
        # Удаляем полностью пустые столбцы и заполняем пропуски нулями
        return binary.dropna(how='all', axis='columns').fillna(0)
    
    def __format_binary(self, binary: pd.DataFrame):
        """
        Внутренний метод форматирования бинарной матрицы.
        
        Параметры:
        ----------
        binary : pd.DataFrame
            Бинарная матрица параметров
                    
        Возвращает:
        -----------
        pd.DataFrame
            Отформатированная бинарная матрица 
        """

        # Заполняем пропуски нулями и сбрасываем индекс
        formatted_binary = binary.fillna(0).astype('int64').reset_index()
        return formatted_binary
    
    def __markup_defects(self, binary: pd.DataFrame, data_events: pd.DataFrame, 
                      days_before_defect):
        # Добавляем столбцы для разметки дней, которые считаем днями с дефектами
        # Для конуса - это несколько дней непосредственно до дня обнаружения
        if days_before_defect > 0:
            detections = data_events[data_events > 0].dropna(how='all').index.to_frame()

            def get_markup_dates(detect_date):
                return pd.date_range(end=detect_date, inclusive='left', periods=days_before_defect + 1, freq='d').to_list()

            markup = pd.DataFrame(detections[self.__date_column].apply(get_markup_dates).to_list(), index=detections.index.get_level_values(0))
            markup = markup.stack()
            markup = markup.droplevel(1)
            markup = markup.to_frame()
            markup = markup.set_index(markup.columns.to_list()[0], append=True)

            present_indexes = binary.index.isin(markup.index)
           
            binary.loc[present_indexes.tolist(), 'defect_markup'] = 1

            # дни обаружения дефектов удаляем
            detection_day_indexes = binary.index.isin(detections.index)
            binary = binary.drop(binary.loc[detection_day_indexes.tolist()].index)
        
        return binary


if __name__ == '__main__':
    df_saz = pd.read_csv('resultall.csv', parse_dates=['Дата'])
    df_saz = arl_data.Data.fix_initial_frame(df_saz, 0)

    m = ArlBinaryMatrix()
   #тест по гистограммам
    m.create_model(df_saz, 'Nomer elektrolizera', '', BinarizationType.HISTOGRAMS, 'Konus (sht)')
    # print('\n\n\t\t\t ---------------Hists-----------------')
    # print('bounds', len(m.boundaries), m.boundaries.columns)
    m.transform(df_saz, 'Konus (sht)', 'Nomer elektrolizera', '')
    # # print('default bin', len(m.binary), m.binary.columns)
    # m.transform(df_saz, 'Konus (sht)', 'Nomer elektrolizera', '', anomalies_only=True)
    # print('anom only bin', len(m.binary), m.binary.columns)
    # # m.transform(df_saz, 'Konus (sht)', 'Nomer elektrolizera', '', keep_nan=False,anomalies_only=True)
    # # print('nan false + anom only bin', len(m.binary), m.binary.columns)

    #
    # #тест по станартному отклонению
    # m.create_model(df_saz, 'Nomer elektrolizera', '', BinarizationType.STDDEV, 'Konus (sht)', ind_type=False)
    # print('\n\n\t\t\t ---------------STDDEV common-----------------')
    # print('bounds', len(m.boundaries), m.boundaries.columns)
    #
    # m.transform(df_saz, 'Konus (sht)', 'Nomer elektrolizera', '')
    # print('default bin', len(m.binary), m.binary.columns)
    # m.transform(df_saz, 'Konus (sht)', 'Nomer elektrolizera', '', anomalies_only=True)
    # print('anom only bin', len(m.binary), m.binary.columns)
    # m.transform(df_saz, 'Konus (sht)', 'Nomer elektrolizera', '', keep_nan=False, anomalies_only=True)
    # print('nan false + anom only bin', len(m.binary), m.binary.columns)
    #
    # #тест по станартному отклонению индивидуальному
    # m.create_model(df_saz, 'Nomer elektrolizera', '', BinarizationType.STDDEV, 'Konus (sht)')
    # print('\n\n\t\t\t ---------------STDDEV ind-----------------')
    # print('bounds', len(m.boundaries), m.boundaries.columns)
    #
    # m.transform(df_saz, 'Konus (sht)', 'Nomer elektrolizera', '')
    # print('default bin', len(m.binary), m.binary.columns)
    # m.transform(df_saz, 'Konus (sht)', 'Nomer elektrolizera', '', anomalies_only=True)
    # print('anom only bin', len(m.binary), m.binary.columns)
    # m.transform(df_saz, 'Konus (sht)', 'Nomer elektrolizera', '', keep_nan=False, anomalies_only=True)
    # print('nan false + anom only bin', len(m.binary), m.binary.columns)
    #
    # #тест по квартилям
    # m.create_model(df_saz, 'Nomer elektrolizera', '', BinarizationType.QUARTILES, 'Konus (sht)', ind_type=False)
    # print('\n\n\t\t\t ---------------QUART common-----------------')
    # print('bounds', len(m.boundaries), m.boundaries.columns)
    #
    # m.transform(df_saz, 'Konus (sht)', 'Nomer elektrolizera', '')
    # print('default bin', len(m.binary), m.binary.columns)
    # m.transform(df_saz, 'Konus (sht)', 'Nomer elektrolizera', '', anomalies_only=True)
    # print('anom only bin', len(m.binary), m.binary.columns)
    # m.transform(df_saz, 'Konus (sht)', 'Nomer elektrolizera', '', keep_nan=False, anomalies_only=True)
    # print('nan false + anom only bin', len(m.binary), m.binary.columns)
    #
    # #тест по квартилям индивидуально
    # m.create_model(df_saz, 'Nomer elektrolizera', '', BinarizationType.QUARTILES, 'Konus (sht)')
    # print('\n\n\t\t\t ---------------QUART ind----------------')
    # print('bounds', len(m.boundaries), m.boundaries.columns)
    #
    # m.transform(df_saz, 'Konus (sht)', 'Nomer elektrolizera', '')
    # print('default bin', len(m.binary), m.binary.columns)
    # m.transform(df_saz, 'Konus (sht)', 'Nomer elektrolizera', '', anomalies_only=True)
    # print('anom only bin', len(m.binary), m.binary.columns)
    # m.transform(df_saz, 'Konus (sht)', 'Nomer elektrolizera', '', keep_nan=False, anomalies_only=True)
    # print('nan false + anom only bin', len(m.binary), m.binary.columns)