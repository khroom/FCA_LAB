import numpy as np
import pandas as pd
import math
import datetime
from enum import Enum
import joblib
import arl_data


class BinarizationType(Enum):
    STDDEV = 0
    QUARTILES = 1
    HISTOGRAMS = 2


class ArlBinaryMatrix:
    def create_model(self, df: pd.DataFrame, obj_column: str, parse_date: str, bin_type: BinarizationType, defect = None, ind_type: bool = True, days_before_defect = 1):
        """
        Разбивает исходные значения параметров на диапазоны в соответствии с выбранным методом.
        Полученные диапазоны(модель) заносятся в атрибут класса boundaries
        :param df: Фрейм исходных данных
        :param defect: Наименование целевого параметра или список целевых параметров, необходим только для бинаризации по гистограммам
        :param obj_column: Столбец с номером электролизера
        :param parse_date: Столбец с датой
        :param bin_type: Тип бинаризации (стандартный разброс, квартили, гистограммы)
        :param ind_type: Тип бинаризации (индивидуально для каждого электролизера или общая)
        :param days_before_defect: Столбец бинарной матрицы для дня до конуса (days_before_defect=1 - один день до конуса)
        """
        self.__obj_column = obj_column
        self.bin_type = bin_type
        self.ind_type = ind_type
        #проверка, нужный ли индекс у df
        if [obj_column, parse_date] == df.index.names:
            data_params = df
        else:
            data_params = df.reset_index()
            if (obj_column in data_params.columns) & (parse_date in data_params.columns):
                data_params = data_params.set_index([obj_column, parse_date])
        if not isinstance(defect, list):
            defect = [defect]
        data_events = df[defect]
        data_params = df.drop(defect, axis=1)
        if bin_type == BinarizationType.STDDEV:
            if ind_type:
                stats = data_params.groupby(obj_column).agg([np.min, np.mean, np.std, np.max])
            else:
                stats = data_params.agg([np.min, np.mean, np.std, np.max])
                stats = pd.concat([stats], keys=['0'], names=['all']).unstack()
            self.boundaries = self.__get_boundaries_by_stat(stats)

        if bin_type == BinarizationType.QUARTILES:
            if ind_type:
                stats = pd.DataFrame()
                for group, data in data_params.groupby(obj_column):
                    stats = stats.append(pd.concat([data.describe()], keys=[group], names=[obj_column]))
                stats = stats.unstack()
            else:
                stats = data_params.describe()
                stats = pd.concat([stats], keys=['0'], names=['all']).unstack()
            self.boundaries = self.__get_boundaries_by_quartiles(stats)

        if bin_type == BinarizationType.HISTOGRAMS:
            if ind_type:
                self.boundaries = pd.DataFrame()
                for group, data in data_params.groupby(obj_column):
                    self.boundaries = self.boundaries.append(pd.concat([self.__get_boundaries_by_hist(data, data_events.loc[data.index], day_before=days_before_defect)], keys=[group], names=[obj_column]))
                self.boundaries.index = self.boundaries.index.droplevel(1)
            else:
                self.boundaries = self.__get_boundaries_by_hist(data_params, data_events, day_before=days_before_defect)
        self.binary = None

    def load_model(self, file: str):
        """
        Загрузка модели из файла в атрибуты класса
        :param file: имя файла
        :return:
        """
        if '.' in file:
            model = joblib.load(file)
        else:
            model = joblib.load(file + ".joblib")
        self.boundaries = model.boundaries
        self.bin_type = model.bin_type
        self.ind_type = model.ind_type
        self.binary = None

    def save_model(self, file: str):
        """
        Выгрузка модели из класса в файл
        :param file: имя файла
        :return:
        """
        if self.boundaries is not None:
            model = self
            model.binary = None
            if '.' in file:
                joblib.dump(self, file)
            else:
                joblib.dump(self, file + ".joblib")
        else:
            raise Exception("No binary model")

    def transform(self, df: pd.DataFrame, defect, obj_column: str, parse_date: str, keep_nan=True, days_before_defect=1, anomalies_only = False):
        """
        Расчет бинарной матрицы по модели
        :param df: фрейм параметров
        :param defect: Целевой параметр
        :param obj_column: Столбец с номером электролизера
        :param parse_date: Столбец с датой
        :param keep_nan: Будет ли бинарная матрица содержать стобцы для NaN значений параметров
        :param days_before_defect:
        :return: бинарная матрица
        """
        if self.boundaries is None:
            raise Exception("No binary model")
        # проверка, нужный ли индекс у df
        if [obj_column, parse_date] == df.index.names:
            data_params = df
        else:
            data_params = df.reset_index()
            if (obj_column in data_params.columns) & (parse_date in data_params.columns):
                data_params = data_params.set_index([obj_column, parse_date])
        if not isinstance(defect, list):
            defect = [defect]
        data_events = df[defect]
        data_params = df.drop(defect, axis=1)
        cross_cols = list(set(data_params.columns).intersection(self.boundaries.columns.get_level_values(0)))
        full_df = self.__concat_data_boundaries(data_params[cross_cols], self.boundaries[cross_cols])

        if self.bin_type == BinarizationType.STDDEV:
            classified_data = self.__classify_by_std(full_df)
            binary = self.__get_binary(classified_data)
            if anomalies_only:
                normal = []
                for c in binary.columns:
                    if c[-2:] == '_1':
                        normal.append(c)
                binary = binary.drop(normal, axis='columns')
            if not keep_nan:
                binary = self.__drop_nan(binary)

        if self.bin_type == BinarizationType.QUARTILES:
            classified_data = self.__classify_by_quartiles(full_df)
            binary = self.__get_binary(classified_data)
            if anomalies_only:
                normal = []
                for c in binary.columns:
                    if (c[-2:] == '_1') | (c[-2:] == '_2'):
                        normal.append(c)
                binary = binary.drop(normal, axis='columns')
            if not keep_nan:
                binary = self.__drop_nan(binary)

        if self.bin_type == BinarizationType.HISTOGRAMS:
            classified_data = self.__classify_by_hist(full_df)
            binary = self.__get_binary(classified_data)
            if anomalies_only & (len(self.boundaries) == 1):
                normal = set()
                for p in self.boundaries.columns.get_level_values(0).unique():
                    for c in self.boundaries[p].columns:
                        if c[1] == 'clean':
                            col = p + '_' + str(c[0])
                            if col in binary.columns:
                                normal.add(col)
                        if c[1] == 'defect':
                            col = p + '_' + str(c[0]+1)
                            if col in binary.columns:
                                normal.add(col)
                binary = binary.drop(list(normal), axis='columns')
            if not keep_nan:
                binary = self.__drop_nan(binary)

        self.binary = self.__format_binary(binary, data_events, days_before_defect)
        return self.binary

    def __drop_nan (self, df: pd.DataFrame):
        nan_cols = []
        for c in df.columns:
            if c[-4:] == '_NaN':
                nan_cols.append(c)
        return df.drop(nan_cols, axis='columns')

    def __get_boundaries_by_stat(self, stats: pd.DataFrame):
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

    def __get_boundaries_by_quartiles(self, stats: pd.DataFrame):
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

    def __get_boundaries_by_hist(self, df: pd.DataFrame, data_events: pd.DataFrame, day_before=1, interval_width=5,
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
            period_dates = pd.date_range(end=index[1] - datetime.timedelta(days=day_before), periods=3)
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
            i = 0
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
                            boundaries[(c, (i,leading))] = index
                            start = index
                            leading = group
                            i += 1
        boundaries.columns = pd.MultiIndex.from_tuples(boundaries.columns)
        boundaries.index.name = 'all'
        return boundaries

    def decode(self):
        """
        Формирование таблицы интервалов по границам
        :param rule:
        :return:
        """
        test = {'Vremia pitaniia po bazovoi ustavke_0', 'Vremia APG v ruch.rezh._0', 'Kol-vo doz APG v nedopitke_3', 'Vremia otkl. APG_0', 'El-lit: temp-ra_3', 'Tsel. napr. el-ra_0', 'Vremia zapreta APG_0', 'Doza glinozema_0', 'El-lit: KO_3', 'Cr. vremennaia dobavka_0', 'AlF3: ustavka pitaniia_1', 'RMPR: koeff._0', 'Cr. dobavka po shumam_0', 'Ustavka APG_0', 'Upravlenie: temp-ra el-ta_3', 'RMPR: vremia zapreta_0', 'El-lit: CaF2_0', 'Tsel. temp. el-ta_0', 'Napr. zad._0', 'El-lit: MgF2_0', 'Kol-vo doz APG v teste_3', 'Shum_0'}
        ranges = pd.concat([self.boundaries], keys=['all'], axis=1).sort_index(axis='index').sort_index(axis='columns')
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
        t = ranges.loc[6.0].swaplevel(0,1).loc[list(test)]
        t.index = t.index.droplevel(1)
        return t

    def __concat_data_boundaries(self, data: pd.DataFrame, boundaries: pd.DataFrame):
        """
        Совмещение исходных параметров и границ, установленных с помощью выбранного метода бинаризации
        :param data: Фрейм исходных параметров
        :param boundaries: Фрейм границ
        :return:
        """
        df = pd.concat([data], axis='columns', keys=['value'])
        df.columns = df.columns.swaplevel(0, 1)
        if boundaries.index.name == self.__obj_column:
            return df.join(boundaries)
        else:
            df = pd.concat([df], keys=['0'], names=['all'])
            df = df.join(boundaries)
            df.index = df.index.droplevel(0)
            return df

    def __classify_by_std(self, df: pd.DataFrame):
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

    def __classify_by_quartiles(self, df: pd.DataFrame):
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

    def __classify_by_hist(self, df: pd.DataFrame):
        """
        Классификация значений параметров по пересечениям гистограмм выборок "чистых" и "с нарушениями".
        Разное количество интевалов для параметров. Классификации подлежат только параметры имеющие два и более интервалов.
        Правая граничнае точка каждого интервала включена в интервал.
        :param df: Фрейм исходных параметров, содержащий границы интервалов
        :return:
        """
        classified = pd.DataFrame(index=df.index)
        for c in df.columns.get_level_values(0).unique():
            df_column = df[c]
            for group, data in df_column.groupby(self.__obj_column):
                dfc = data.dropna(axis=1, how='all')
                i = 0
                if len(dfc.columns) > 1:
                    if (i, 'defect') in dfc.columns:
                        classified.loc[dfc[dfc['value'] <= dfc[(i, 'defect')]].index, c] = i
                    if (i, 'clean') in dfc.columns:
                        classified.loc[dfc[dfc['value'] <= dfc[(i, 'clean')]].index, c] = i
                    i += 1
                    while i <= dfc.columns[-1][0]:
                        if (i, 'defect') in dfc.columns:
                            classified.loc[dfc[(dfc['value'] > dfc[(i-1, 'clean')]) & (dfc['value'] <= dfc[(i, 'defect')])].index, c] = i
                        if (i, 'clean') in dfc.columns:
                            classified.loc[dfc[(dfc['value'] > dfc[(i-1, 'defect')]) & (dfc['value'] <= dfc[(i, 'clean')])].index, c] = i
                        i += 1
                    #последние интервалы. Если последняя граница 'clean', значит, последний интервал - 'defect'
                    if (i-1, 'clean') in dfc.columns:
                        classified.loc[dfc[(dfc['value'] > dfc[(i-1, 'clean')])].index, c] = i
                    if (i-1, 'defect') in dfc.columns:
                        classified.loc[dfc[(dfc['value'] > dfc[(i-1, 'defect')])].index, c] = i
        return classified

    def __get_binary(self, df: pd.DataFrame):
        """
        Формирование бинарной матрицы
        :param df: Фрейм, в тотором значениями параметров являются номера групп, к которым принадлежат исходные значение параметров.
        :return:
        """
        data = df
        binary = pd.DataFrame(index=data.index)
        for c in data.columns:
            i = 0
            while i <= data[c].max():
                binary.loc[data[c] == i, c + '_' + str(int(i))] = 1
                i += 1
            binary.loc[data[c].isna(), c + '_NaN'] = 1
        return binary.dropna(how='all', axis='columns').fillna(0)

    def __format_binary(self, binary: pd.DataFrame(), data_events: pd.DataFrame(), days_before_defect):
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