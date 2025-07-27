# Импорт необходимых библиотек
import arl_fca_lab  # Библиотека для анализа формальных понятий
import arl_binarization  # Библиотека для бинаризации данных
import arl_data  # Библиотека для предобработки данных
import pandas as pd  # Библиотека для работы с табличными данными
import numpy  # Библиотека для численных операций
import matplotlib.pyplot as plt  # Библиотека для визуализации данных

# Основной блок выполнения программы
if __name__ == '__main__':
    print('Загрузка исходных данных')

    date_name = 'DATE_TRUNC'  # Название столбца с датами
    obj_name = 'POT_ID'  # Название столбца с номерами электролизеров
    defect_name = 'Вид  нарушения_количество'  # Название целевого параметра (дефекта)
    dif_date = '2025-01-01'  # Дата разделения на обучающую и тестовую выборки

    el_list = [1, 2, 3, 4, 5, 6, 7, 8]  # Список электролизеров для анализа

    # Загрузка данных из CSV файла
    df = pd.read_csv('../result_SAZ25.csv', parse_dates=[date_name])

    # Предобработка данных
    # add_df = df[defect_name]  # Временное сохранение столбца violation
    df = df.drop(['№ анода','возраст на момент обнаружения', 'ROOM_ID', 'SHOP_ID', 'Вид  нарушения_наименование'], axis=1)  # Удаление лишних столбцов

    # Разделение на обучающую и тестовую выборки
    train_df = df[(df[date_name] > '2024-10-01')&(df[date_name] < dif_date)&(df[obj_name].isin(el_list))]
    test_df = df[(df[date_name] < '2025-01-30')&(df[date_name] >= dif_date)&(df[obj_name].isin(el_list))]

    print('Выполнено')

    # Разделение на обучающую и тестовую выборки
    train_df = df[(df[date_name] > '2024-10-01')&(df[date_name] < dif_date)&(df[obj_name].isin(el_list))]
    test_df = df[(df[date_name] < '2025-01-30')&(df[date_name] >= dif_date)&(df[obj_name].isin(el_list))]

    print('Подготовка')
    # Установка мультииндекса (номер электролизера + дата)
    train_df.set_index([obj_name, date_name], inplace=True)
    test_df.set_index([obj_name, date_name], inplace=True)

    # Удаление из тестовой выборки столбцов, которых нет в обучающей
    test_df = test_df.drop(list(set(test_df.columns).difference(set(train_df.columns))), axis=1)
    print('Выполнено')
    
    print('Расчет модели')

    # Создание модели анализа формальных понятий с параметрами:
    # - тип бинаризации: по стандартному отклонению (STDDEV)
    # - индивидуальный тип: False (строится общая модель для всех электролизеров)
    # - keep_nan: False (не сохранять пропущенные значения)
    model = arl_fca_lab.arl_fca_lab(bin_type=arl_binarization.BinarizationType.STDDEV, ind_type=False, keep_nan=True)
    
    # Обучение модели на обучающей выборке
    model.fit_model(train_df, defect_name, obj_name, date_name, anomalies_only=True)

    print('Выполнено')
    print('Подготовка тестового датасета')

    # Преобразование тестовых данных с использованием полученной модели бинаризации
    test_matrix = model.bin_matrix.transform(
        test_df, 
        model.defect, 
        obj_name, 
        date_name,
        model.keep_nan, 
        model.days_before_defect, 
        model.anomalies_only
    )
    print('Выполнено')
    print('Расчет оценок')

    # Создание DataFrame для хранения результатов прогнозирования
    test_results = pd.DataFrame(index=test_matrix.index, columns=['Object', 'Prediction', 'Fact', 'Rule'])

    # Прогнозирование для каждой строки тестовых данных
    for i in test_matrix.index:
        if model.ind_type:
            obj_num = test_matrix.loc[i, obj_name]  # Номер электролизера для индивидуальной модели
        else:
            obj_num = 'all'  # Общая модель
            
        # Получение прогноза
        prediction = model.get_prediction(arl_fca_lab.get_row(test_matrix, i), obj_num)
        
        # Сохранение результатов
        test_results.loc[i, 'Object'] = test_matrix.loc[i, obj_name]
        test_results.loc[i, 'Fact'] = test_matrix.loc[i, '1_day_before']

        if prediction == None:
            # Если правило не найдено
            test_results.loc[i, 'Prediction'] = 0
            test_results.loc[i, 'Rule'] = -1
        else:
            # Если правило найдено
            test_results.loc[i, 'Prediction'] = prediction[0]
            test_results.loc[i, 'Rule'] = prediction[1]
            
        # Вывод информации о текущем прогнозе
        print('Index:', i, ', Object:', test_results.loc[i, 'Object'], 
              ', Predict:', test_results.loc[i, 'Prediction'],
              ', Fact:',test_results.loc[i, 'Fact'])
    
    print('Выполнено')
    # Объединение результатов с исходными данными
    test_results = pd.concat([test_results, test_matrix[[date_name, defect_name, '1_day_before']]], axis=1)
    
    # Расчет метрик качества для разных пороговых значений
    threshold_f1= pd.DataFrame()
    for threshold in numpy.arange(0.01, 0.99, 0.01):
        # Расчет True Positive (TP)
        threshold_f1.loc[threshold, 'TP'] = len(
            test_results[(test_results.Fact == 1)&(test_results.Prediction >= threshold)])
        # Расчет False Positive (FP)
        threshold_f1.loc[threshold, 'FP'] = len(
            test_results[(test_results.Fact == 0)&(test_results.Prediction >= threshold)])
        # Расчет False Negative (FN)
        threshold_f1.loc[threshold, 'FN'] = len(
            test_results[(test_results.Fact == 1) & (test_results.Prediction < threshold)])
        # Расчет True Negative (TN)
        threshold_f1.loc[threshold, 'TN'] = len(
            test_results[(test_results.Fact == 0) & (test_results.Prediction < threshold)])
        # Сумма TN и TP
        threshold_f1.loc[threshold, 'TN_TP'] = threshold_f1.loc[threshold, 'TN']+threshold_f1.loc[threshold, 'TP']
        threshold_f1.loc[threshold, 'Threshold'] = threshold

    # Визуализация результатов
    fig, ax = plt.subplots()
    # График доли истинно положительных результатов
    ax.plot(threshold_f1['Threshold'], threshold_f1['TP'] / len(test_results[test_results['Fact'] == 1]), 'g',
            # График доли ложно положительных результатов
            threshold_f1['Threshold'], threshold_f1['FP'] / len(test_results[(test_results['1_day_before'] == 0)&(test_results['Fact'] == 0)]), 'r')

    # Настройка отображения графика
    plt.xlabel("Пороговое значение")
    plt.ylabel("Доля")
    plt.legend(['Истинно положительные', 'Ложно положительные'])
    # Сохранение графика в файл
    plt.savefig('2025')
    plt.clf()  # Очистка текущей фигуры