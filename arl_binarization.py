import numpy as np
import pandas as pd
import json


def get_binary_stddev(file, individual=True):
    """Создание бинарной матрицы на основе соответствия параметров электролиза нормальным значениям.
    Нормальные значения соответствуют размаху стандратного отклонения
    значения параметра меньше нормы переносятся в столбец с именем параметра плюс _0, в норме: _1, больше нормы: _2

    :param file: файл .csv
        Файл исходных данных
    :param individual
        Если individual=True диапазон нормальных значений рассчитывается для каждого электролизера отдельно,
        иначе диапазон нормальных значений рассчитывается по данным всех электролизеров вместе
    :return: pandas.DataFrame
        Бинарная матрица в форме фрейма данных
    """
    data_events = pd.read_csv(file, sep=';', decimal=',', encoding='cp1251', index_col=['VANNA', 'RDATE'],
                              parse_dates=['RDATE'],
                              usecols=['VANNA', 'RDATE', 'KOL_PODDERNYTIH_ANODOV', 'AE', 'KONUSOV'])
    data_params = pd.read_csv(file, sep=';', decimal=',', encoding='cp1251', index_col=['VANNA', 'RDATE'],
                              parse_dates=['RDATE']).drop(['KONUSOV', 'AE', 'KOL_PODDERNYTIH_ANODOV'], axis='columns')
    # правка исходных данных
    data_params = correct_haz(data_params)
    # статистические параметры рассчитаны по данным до интерполяции
    if individual:
        statistics = data_params.groupby('VANNA').agg([np.mean, np.std])
    else:
        statistics = data_params.agg([np.mean, np.std])

    data_params = interpolate_with_polynomials(data_params)
    data_params.columns = pd.MultiIndex.from_product([data_params.columns, ['value']])

    if individual:
        data_joined = data_params.join(statistics, on='VANNA')
    else:
        stat_temp = pd.DataFrame(columns=pd.MultiIndex.from_product([statistics.columns, statistics.index]), index=[0])
        stat_temp.index.name = 'all'
        for column in stat_temp.columns:
            stat_temp.loc[0, column] = statistics.loc[column[1], column[0]]
        data_params = pd.concat([data_params], keys=[0], names=['all'])
        data_joined = data_params.join(stat_temp, on='all')
        data_joined.index = data_joined.index.droplevel(0)

    data_joined.sort_index(axis='columns', inplace=True)
    data_joined.sort_index(axis='index', inplace=True)
    # классификация по соотвествию статистической норме
    data_classified = pd.DataFrame(columns=data_joined.columns, index=data_joined.index)
    for param in data_joined.columns.get_level_values(0).unique():
        data_classified.loc[data_joined[(param, 'value')] < data_joined[(param, 'mean')] - data_joined[(param, 'std')], [(param, 'value')]] = 0
        data_classified.loc[data_joined[(param, 'value')].between(data_joined[(param, 'mean')] - data_joined[(param, 'std')], data_joined[(param, 'mean')] + data_joined[(param, 'std')]), [(param, 'value')]] = 1
        data_classified.loc[data_joined[(param, 'value')] > data_joined[(param, 'mean')] + data_joined[(param, 'std')], [(param, 'value')]] = 2

    data_extremes = data_classified.drop(columns=['mean', 'std'], axis='columns', level=1)
    data_extremes.columns = data_extremes.columns.droplevel(1)

    # бинаризация
    data_extremes.dropna(inplace=True)
    data_columns = data_extremes.columns
    for column in data_columns:
        data_extremes.loc[data_extremes[column] == 0, column + '_0'] = 1
        data_extremes.loc[data_extremes[column] == 1, column + '_1'] = 1
        data_extremes.loc[data_extremes[column] == 2, column + '_2'] = 1

    # бинаризация технологических нарущений
    data_events.fillna(0, inplace=True)
    data_events[data_events > 0] = 1

    # Финальное форматирование бинарной матрицы
    binary = data_extremes.drop(axis='columns', columns=data_columns).join(data_events)
    binary = binary.fillna(0).astype('int64').reset_index()

    return binary


def get_binary_quartile(file):
    """Создание бинарной матрицы на основе разбиения значений параметров на квартили.
        Столбец с именем параметра плюс _0 - первый квартиль, _1 - второй квартиль,  _2 - третий квартиль, _3 - четвертый квартиль.

        :param file: файл .csv
            Файл исходных данных
        :return: pandas.DataFrame
            Бинарная матрица в форме фрейма данных
    """
    data_events = pd.read_csv(file, sep=';', decimal=',', encoding='cp1251', index_col=['VANNA', 'RDATE'],
                              parse_dates=['RDATE'],
                              usecols=['VANNA', 'RDATE', 'KOL_PODDERNYTIH_ANODOV', 'AE', 'KONUSOV'])
    data_params = pd.read_csv(file, sep=';', decimal=',', encoding='cp1251', index_col=['VANNA', 'RDATE'],
                              parse_dates=['RDATE']).drop(['KONUSOV', 'AE', 'KOL_PODDERNYTIH_ANODOV'], axis='columns')
    # правка исходных данных
    data_params = correct_haz(data_params)
    statistics = data_params.agg([np.min, np.max])
    data_params = interpolate_with_polynomials(data_params)

    quartiles = pd.DataFrame(columns=pd.MultiIndex.from_product([statistics.columns, ['q1', 'q2', 'q3']]), index=[0])
    quartiles.index.name = 'all'
    # расчет квартилей
    for column in statistics.columns:
        q = (statistics.loc['amax', column] - statistics.loc['amin', column]) / 4
        quartiles.loc[0, (column, 'q1')] = statistics.loc['amin', column] + q
        quartiles.loc[0, (column, 'q2')] = statistics.loc['amin', column] + q * 2
        quartiles.loc[0, (column, 'q3')] = statistics.loc['amin', column] + q * 3

    data_params.columns = pd.MultiIndex.from_product([data_params.columns, ['value']])
    data_params = pd.concat([data_params], keys=[0], names=['all'])
    data_params = data_params.join(quartiles, on='all')

    data_params.sort_index(axis='columns', inplace=True)
    data_params.sort_index(axis='index', inplace=True)
    data_join = pd.DataFrame(columns=data_params.columns, index=data_params.index)

    # классификация по вхождению в квартили
    for param in data_params.columns.get_level_values(0).unique():
        data_join.loc[data_params[(param, 'value')] <= data_params[(param, 'q1')], [(param, 'value')]] = 0
        data_join.loc[data_params[(param, 'value')].between(data_params[(param, 'q1')], data_params[(param, 'q2')]), [
            (param, 'value')]] = 1
        data_join.loc[data_params[(param, 'value')].between(data_params[(param, 'q2')], data_params[(param, 'q3')]), [
            (param, 'value')]] = 2
        data_join.loc[data_params[(param, 'value')] > data_params[(param, 'q3')], [(param, 'value')]] = 3

    data_join = data_join.drop(['q1', 'q2', 'q3'], axis=1, level=1)
    data_join.index = data_join.index.droplevel(0)
    data_join.columns = data_join.columns.droplevel(1)

    data_join.dropna(inplace=True)
    data_columns = data_join.columns
    for column in data_columns:
        data_join.loc[data_join[column] == 0, column + '_0'] = 1
        data_join.loc[data_join[column] == 1, column + '_1'] = 1
        data_join.loc[data_join[column] == 2, column + '_2'] = 1
        data_join.loc[data_join[column] == 3, column + '_3'] = 1

    # бинаризация технологических нарущений
    data_events.fillna(0, inplace=True)
    data_events[data_events > 0] = 1

    # Финальное форматирование бинарной матрицы
    binary = data_join.drop(axis='columns', columns=data_columns).join(data_events)
    binary = binary.fillna(0).astype('int64').reset_index()
    return binary


def correct_haz(haz_data_frame):
    """Корректировка исходных данных ХАЗа:
    удаление пустых и текстовых столбцов: 'KOL_GLINOZEMA_TCHEREZ_APG', 'PITANIE','KOL_RABOT_ELECTROLIZEROV'
    заполнение нулями пропусков в столбцах: 'KOL_DOZ_APG_V_TESTE', 'RMPR', 'RMPR_DLITELNOST_VIRA', 'RMPR_DLITELNOST_MAINA','RMPR_KOL_VIRA', 'FTORID_ALUMINIA_VES_DOZI'
    данные столбцов 'RMPR_KOL_MAINA' и 'RMPR_KOL_MAINA_10_KORPUS' объединяются в первый столбец, второй удаляется

    :param haz_data_frame: pandas.DataFrame
        Фрейм данных, который подлежит корректировке
    :return: pandas.DataFrame
        Скорректированный фрейм данных
    """
    columns_for_delete = ['KOL_GLINOZEMA_TCHEREZ_APG', 'PITANIE', 'KOL_RABOT_ELECTROLIZEROV']
    columns_for_nan_replace = ['KOL_DOZ_APG_V_TESTE', 'RMPR', 'RMPR_DLITELNOST_VIRA', 'RMPR_DLITELNOST_MAINA','RMPR_KOL_VIRA', 'FTORID_ALUMINIA_VES_DOZI']
    haz_data_frame.drop(columns_for_delete, axis='columns', inplace=True)
    haz_data_frame.loc[:, columns_for_nan_replace] = haz_data_frame.loc[:, columns_for_nan_replace].fillna(0)
    haz_data_frame.loc[:, 'RMPR_KOL_MAINA'] = (haz_data_frame['RMPR_KOL_MAINA_10_KORPUS'].fillna(0) + haz_data_frame['RMPR_KOL_MAINA'].fillna(0))
    haz_data_frame.drop(['RMPR_KOL_MAINA_10_KORPUS'], axis='columns', inplace=True)
    return haz_data_frame


def interpolate_with_polynomials(data_frame):
    """Интерполяция данных методом полиномов пятого порядка
    применяется к столбцам 'DLITELNOST_VYLIVKI','OTNOSHENIE_SKOROSTI','TEMPERATURA_ELECTROLITA','UROVEN_METALLA','ELECTROLIT','CAF2','MGF2', 'KO','FTORID_PITANIE'

    :param data_frame: pandas.DataFrame
        Фрейм данных, указанные столбцы в котором интерполируются
    :return: data_frame: pandas.DataFrame
        Фрейм данных, указанные столбцы интерполированы, остальные неизменны
    """
    columns_for_interpolation = ['DLITELNOST_VYLIVKI', 'OTNOSHENIE_SKOROSTI', 'TEMPERATURA_ELECTROLITA',
                                 'UROVEN_METALLA', 'ELECTROLIT', 'CAF2', 'MGF2', 'KO', 'FTORID_PITANIE']
    #сортировка обязательна для полиномиальной интерполяции
    data_frame.sort_index(axis='columns', inplace=True)
    data_frame.sort_index(axis='index', inplace=True)
    interpolated = pd.DataFrame()
    for vanna in data_frame.index.get_level_values(0).unique():
        interpolated = interpolated.append(pd.concat([data_frame.loc[vanna, columns_for_interpolation].interpolate(method='polynomial', order=5).round(decimals=2)], keys=[vanna], names=['VANNA']))
    data_frame.drop(columns_for_interpolation, axis='columns', inplace=True)
    data_frame = data_frame.join(interpolated, on=['VANNA', 'RDATE'])
    return data_frame


def replace_objects(lattice):
    """Меняет простые суррогатные идентификаторы объектов в каждом концепте решетки
     на пары вида (номер электролизера, дата)
     в соответствии со справочником объекстов lattice.context_objects

    :param lattice: pca_lattice
        Исходная решетка для корректировки
    :return: none
    """
    for concept in lattice.concepts:
        new_a_list = set()
        for i in lattice.context_objects.loc[concept['A']].values:
            new_a_list.add(tuple(i))
        concept['A'] = new_a_list


def print_items_count(lattice):
    """Вывод в консоль статистики по решетке:
    1. количество атрибутов и количество концептов с таким объемом атрибутов,
    2. концепты с наибольшим объемом,
    3. концепты с наименьшим объемом.

    :param lattice: pca_lattice
        Решетка
    :return: none
    """
    concept_items_count = pd.DataFrame(columns=['Objects', 'Attributes'])
    for concept in lattice.concepts:
        concept_items_count = concept_items_count.append({'Objects': len(concept['A']), 'Attributes': len(concept['B'])}, ignore_index=True)
    concept_items_count = concept_items_count.sort_values(by='Objects', ascending=False)
    print('\nКоличество атрибутов и количество концептов с таким объемом атрибутов:')
    print(concept_items_count.groupby('Attributes').count())
    print('\nКонцепты с наибольшим объемом:')
    print(concept_items_count.head(10))
    print('\nКонцепты с наименьшим объемом:')
    print(concept_items_count.tail(10))


def get_concepts_by_attr(lattice, attribute):
    """Возвращает подмножество концептов решетки, содержанщих атрибут attribute

    :param lattice: pca_lattice
        Решетка, в которой ведется поиск
    :param attribute: str
        Атрибут в стоковом формате
    :return: list
        Подмножество pca_lattice.concepts в виде списка
    """
    attr_lattice = []
    for concept in lattice.concepts:
        if attribute in concept['B']:
            attr_lattice.append(concept)
    return attr_lattice
