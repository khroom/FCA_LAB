import numpy as np
import pandas as pd
import math
import datetime
from enum import Enum


class BinarizationType(Enum):
    STDDEV_INDIVIDUAL = 0
    STDDEV_COMMON = 1
    QUARTILES = 2
    HISTOGRAMS = 3


class arl_binary_matrix:
    def __init__(self, df: pd.DataFrame, bin_type: BinarizationType, defect: list, cell_column, nan_cols=True,
                 days_before_defect=1):
        """
        Разбивает исходные значения параметров на группы в соответствии с выбранным методом.
        Строит бинарную матрицу (self.binary) по вхождению значений параметров в группы.
        Предоставляет интервалы значений для групп (self.ranges).
        :param df: Фрейм исходных данных. Предполагается, что фрейм имеет сложный индекс, содержащий номер электролизера
         и дату измерения.
        :param bin_type: Тип бинаризации (стандартное отклонение параметров, индивидуальное стандартное отклонение
        для электролизеров, квартили, пересечения гистограмм выборок "чистых" и "с нарушениями")
        :param defect: Список, содержащий целевой параметр(ы)
        :param cell_column: Наименование электролизера в индексе
        :param nan_cols: Будет ли бинарная матрица содержать стобцы для NaN значений параметров
        :param days_before_defect: Столбец бинарной матрицы для дня до конуса (days_before_defect=1 - один день до конуса)
        """
        self.cell_column = cell_column
        data_events = df[defect]
        data_params = df.drop(defect, axis=1)

        if bin_type == BinarizationType.STDDEV_INDIVIDUAL:
            stats = data_params.groupby(cell_column).agg([np.min, np.mean, np.std, np.max])
            boundaries = self.get_boundaries_by_stat(stats)
            full_df = self.concat_data_boundaries(data_params, boundaries)
            classified_data = self.classify_by_std(full_df)

        if bin_type == BinarizationType.STDDEV_COMMON:
            stats = data_params.agg([np.min, np.mean, np.std, np.max])
            stats = pd.concat([stats], keys=['0'], names=['all']).unstack()
            boundaries = self.get_boundaries_by_stat(stats)
            full_df = self.concat_data_boundaries(data_params, boundaries)
            classified_data = self.classify_by_std(full_df)

        if bin_type == BinarizationType.QUARTILES:
            stats = data_params.describe().loc[['min', '25%', '50%', '75%', 'max']]
            stats = pd.concat([stats], keys=['0'], names=['all']).unstack()
            boundaries = self.get_boundaries_by_quartiles(stats)
            full_df = self.concat_data_boundaries(data_params, boundaries)
            classified_data = self.classify_by_quartiles(full_df)

        if bin_type == BinarizationType.HISTOGRAMS:
            boundaries = self.get_boundaries_by_hist(data_params, data_events, day_before=days_before_defect)
            full_df = self.concat_data_boundaries(data_params, boundaries)
            classified_data = self.classify_by_hist(full_df)

        binary = self.get_binary(classified_data, nan_cols)
        self.binary = self.format_binary(binary, data_events, days_before_defect)
        self.ranges = self.get_ranges(boundaries)

    def get_boundaries_by_stat(self, stats: pd.DataFrame):
        """
        Расчет границ групп при разделении по стандартному отклонению
        :param stats: Фрейм статистических данных
        :return:
        """
        boundaries = pd.DataFrame(index=stats.index)
        for c in stats.columns.get_level_values(0).unique():
            boundaries[(c, 0)] = stats[(c, 'amin')]
            boundaries[(c, 1)] = stats[(c, 'mean')] - stats[(c, 'std')]
            boundaries[(c, 2)] = stats[(c, 'mean')] + stats[(c, 'std')]
            boundaries[(c, 3)] = stats[(c, 'amax')]
        boundaries.columns = pd.MultiIndex.from_tuples(boundaries.columns)
        return boundaries

    def get_boundaries_by_quartiles(self, stats: pd.DataFrame):
        """
        Расчет границ групп при разделении по квартилям
        :param stats: Фрейм статистических данных
        :return:
        """
        boundaries = pd.DataFrame(index=stats.index)
        for c in stats.columns.get_level_values(0).unique():
            boundaries[(c, 0)] = stats[(c, 'min')]
            boundaries[(c, 1)] = stats[(c, '25%')]
            boundaries[(c, 2)] = stats[(c, '50%')]
            boundaries[(c, 3)] = stats[(c, '75%')]
            boundaries[(c, 4)] = stats[(c, 'max')]
        boundaries.columns = pd.MultiIndex.from_tuples(boundaries.columns)
        return boundaries

    def get_boundaries_by_hist(self, df: pd.DataFrame, data_events: pd.DataFrame, day_before=1, interval_width=5,
                               thres=0.1):
        """
        Расчет границ групп при разделении по пересечениям гистограмм выборок "чистых" и "с нарушениями"
        с использованием метода скользящего среднего
        :param df: Исходный фрейм с данными электролиза
        :param data_events: Фрейм с данными по нарушению (целевым параметром)
        :param day_before: День для отсчета пятидневного периода формирования нрушения, по умелчанию - один день до конуса
        :param interval_width: Ширина интервала для скользящего среднего
        :param thres: Порог чувствительности для разделения групп
        :return:
        """
        #разделение на чистые и с нарушениями
        #нарушение - пять дней от day_before
        defect_period = []
        for index, value in data_events[data_events > 0].dropna(how='all').iterrows():
            period_dates = pd.date_range(end=index[1] - datetime.timedelta(days=day_before), periods=6)
            for d in period_dates:
                if (index[0], d) in df.index:
                    defect_period.append((index[0], d))
        defect_frame = df.loc[defect_period]
        clean_frame = df.drop(defect_period)
        boundaries = pd.DataFrame(index=['0'])
        for c in df.columns:
            if df[c].count() == 0:
                break
            # ширина интревала гистограммы
            step = 3.5 * df[c].std() / math.pow(df[c].count(), 1 / 3)
            bins = []
            clean_hist = []
            defect_hist = []
            cur_value = df[c].min()
            # расчет гитограмм выборок
            while cur_value < df[c].max() + step:
                clean_count = clean_frame.loc[(clean_frame[c] >= cur_value) & (clean_frame[c] < cur_value + step)][
                                  c].count() / clean_frame[c].count()
                defect_count = defect_frame.loc[(defect_frame[c] >= cur_value) & (defect_frame[c] < cur_value + step)][
                                   c].count() / defect_frame[c].count()
                if (clean_count > 0) | (defect_count > 0):
                    bins.append(cur_value)
                    clean_hist.append(clean_count)
                    defect_hist.append(defect_count)
                cur_value = cur_value + step
            hist = pd.DataFrame({'clean': clean_hist, 'defect': defect_hist}, index=bins)
            # минимально значимая разница между выборками на интервале
            delta = hist.max().max() * thres
            leading = ''
            start = hist.iloc[0].name
            end = hist.iloc[hist.count().max() - 1].name
            boundaries[(c, 0)] = start
            i = 1
            for index, row in hist.iloc[1:].iterrows():
                if (row['clean'] >= delta) | (row['defect'] >= delta):
                    # различие между выборками на основе скользящего среднего
                    change = ((hist['clean'] - hist['defect']).loc[:index].tail(interval_width).rolling(
                        interval_width).mean()).loc[index]
                    group = 'defect' if change < 0 else 'clean'
                    # произошла смета доминирующей группы
                    if group != leading:
                        if leading == '':
                            leading = group
                        else:
                            boundaries[(c, i)] = index
                            start = index
                            leading = group
                            i += 1
            boundaries[(c, i)] = end
        boundaries.columns = pd.MultiIndex.from_tuples(boundaries.columns)
        boundaries.index.name = 'all'
        return boundaries

    def get_ranges(self, boundaries: pd.DataFrame):
        """
        Формирование таблицы интервалов по границам
        :param boundaries: Таблица, содержащая границы групп для парамтров
        :return:
        """
        ranges = pd.concat([boundaries], keys=['all'], axis=1).sort_index(axis='index').sort_index(axis='columns')
        for c in ranges.columns.get_level_values(1).unique():
            i = 0
            while i < len(ranges[('all', c)].columns) - 1:
                ranges[(c, c + '_' + str(i), 'min')] = ranges[('all', c, i)]
                ranges[(c, c + '_' + str(i), 'max')] = ranges[('all', c, i + 1)]
                i += 1
        ranges = ranges.drop(['all'], axis=1, level=0)
        ranges = ranges.stack([0, 1])
        ranges = ranges[['min', 'max']]
        if ranges.index.names[0] == 'all':
            ranges.index = ranges.index.droplevel(0)
        return ranges

    def concat_data_boundaries(self, data: pd.DataFrame, boundaries: pd.DataFrame):
        """
        Совмещение исходных параметров и границ, установленных с помощью выбранного метода бинаризации
        :param data: Фрейм исходных параметров
        :param boundaries: Фрейм границ
        :return:
        """
        df = pd.concat([data], axis='columns', keys=['value'])
        df.columns = df.columns.swaplevel(0, 1)
        if boundaries.index.name == self.cell_column:
            return df.join(boundaries)
        else:
            df = pd.concat([df], keys=['0'], names=['all'])
            df = df.join(boundaries)
            df.index = df.index.droplevel(0)
            return df

    def classify_by_std(self, df: pd.DataFrame):
        """
        Классификация значений параметров по стандартному отклонению.
        Три интервала для каждого параметра. Граничные точки центрального интервала включены в центральынй интервал.
        :param df: Фрейм исходных параметров, содержащий границы интервалов
        :return:
        """
        classified = pd.DataFrame(index=df.index)
        for c in df.columns.get_level_values(0).unique():
            classified.loc[df[(c, 'value')] < df[(c, 1)], c] = 0
            classified.loc[df[(c, 'value')].between(df[(c, 1)], df[(c, 2)]), c] = 1
            classified.loc[df[(c, 'value')] > df[(c, 2)], c] = 2
        return classified

    def classify_by_quartiles(self, df: pd.DataFrame):
        """
        Классификация значений параметров по вхождению в квартили.
        Четыре интервала для каждого параметра. Правая граничнае точка каждого интервала включена в интервал.
        :param df: Фрейм исходных параметров, содержащий границы интервалов
        :return:
        """
        classified = pd.DataFrame(index=df.index)
        for c in df.columns.get_level_values(0).unique():
            classified.loc[df[(c, 'value')] <= df[(c, 1)], c] = 0
            classified.loc[(df[(c, 'value')] > df[(c, 1)]) & (df[(c, 'value')] <= df[(c, 2)]), c] = 1
            classified.loc[(df[(c, 'value')] > df[(c, 2)]) & (df[(c, 'value')] <= df[(c, 3)]), c] = 2
            classified.loc[df[(c, 'value')] > df[(c, 3)], c] = 3
        return classified

    def classify_by_hist(self, df: pd.DataFrame):
        """
        Классификация значений параметров по пересечениям гистограмм выборок "чистых" и "с нарушениями".
        Разное количество интевалов для параметров. Классификации подлежат только параметры имеющие два и более интервалов.
        Правая граничнае точка каждого интервала включена в интервал.
        :param df: Фрейм исходных параметров, содержащий границы интервалов
        :return:
        """
        classified = pd.DataFrame(index=df.index)
        for c in df.columns.get_level_values(0).unique():
            i = 0
            if len(df[c].columns) > 3:
                while i < len(df[c].columns) - 2:
                    classified.loc[(df[(c, 'value')] > df[(c, i)]) & (df[(c, 'value')] <= df[(c, i + 1)]), c] = i
                    i += 1
        return classified

    def get_binary(self, df: pd.DataFrame, nan_cols=True):
        """
        Формирование бинарной матрицы
        :param df: Фрейм, в тотором значениями параметров являются номера групп, к которым принадлежат исходные значение параметров.
        :param nan_cols: Будет ли бинарная матрица содержать стобцы для NaN значений параметров
        :return:
        """
        data = df
        if not nan_cols:
            data.dropna(inplace=True)
        binary = pd.DataFrame(index=data.index)
        for c in data.columns:
            i = 0
            while i <= data[c].max():
                binary.loc[data[c] == i, c + '_' + str(int(i))] = 1
                i += 1
            if nan_cols:
                binary.loc[data[c].isna(), c + '_NaN'] = 1
        return binary.dropna(how='all', axis='columns').fillna(0)

    def format_binary(self, binary: pd.DataFrame(), data_events: pd.DataFrame(), days_before_defect):
        """
        Форматирование бинарной матрицы: добавление бинаризованных столбцов целевого параметра и дня до, сброс индекса.
        :param binary: Бинарная матрица
        :param data_events: Фрейм целевого параметра
        :param days_before_defect: Столбец бинарной матрицы для дня до конуса (days_before_defect=1 - один день до конуса)
        :return:
        """
        bin_events = pd.DataFrame(index=data_events.index)
        for c in data_events.columns:
            bin_events.loc[data_events[c] > 0, c] = 1
        binary = binary.join(bin_events)
        if days_before_defect > 0:
            days_before = []
            for index, row in data_events[data_events > 0].dropna(how='all').iterrows():
                d = index[1] - datetime.timedelta(days=days_before_defect)
                if (index[0], d) in data_events.index:
                    days_before.append((index[0], d))
            binary.loc[days_before, str(days_before_defect) + '_day_before'] = 1
        formatted_binary = binary.fillna(0).astype('int64').reset_index()
        return formatted_binary

if __name__ == '__main__':
    df_saz = pd.read_csv('resultall.csv', parse_dates=['Дата'])
    df_saz.drop(['Индекс', 'АЭ: кол-во', 'АЭ: длит.', 'АЭ: напр. ср.', 'Пена: снято с эл-ра', 'Срок службы',
             'Кол-во поддер. анодов', 'АЭ: кол-во локальных', 'Отставание (шт)', 'Другое нарушение (шт)',
             'Нарушение', 'Анод', 'Выливка: отношение скорости', 'Выход по току: Л/О', 'Подина: напр.',
             'Кол-во доз АПГ в руч.реж.', 'Конус (да/нет)'], axis='columns', inplace=True)
    import unidecode
    # замена кириллицы на латинницу
    for column in df_saz.columns:
        df_saz.rename(columns={column: unidecode.unidecode(column).replace('\'', '')}, inplace=True)
    df_saz = df_saz.set_index(['Nomer elektrolizera','Data'])

    bin_hist = arl_binary_matrix(df_saz, BinarizationType.HISTOGRAMS, ['Konus (sht)'], cell_column = 'Nomer elektrolizera')
    bin_hist.binary.to_csv('saz_binary\\SAZ_binary_hist.csv')
    bin_hist.ranges.to_csv('saz_binary\\SAZ_ranges_hist.csv')
