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
    # Загрузка исходных данных из CSV-файла с парсингом даты в столбце 'dt'
    init_df = pd.read_csv('result_19_01-20_11.csv', parse_dates=['dt'])

    # Разделение номеров электролизеров на две группы: 5-й корпус (<6000) и 6-й корпус (>=6000)
    objt_list_5 = [i for i in init_df['Номер электролизера'].unique() if i < 6000]
    objt_list_6 = [i for i in init_df['Номер электролизера'].unique() if i >= 6000]

    # Список параметров, которые будут использоваться в анализе
    params = ['Срок службы', 'Эл-лит: темп-ра', 'Выливка: Л/О', 'РМПР: коэфф.', 'Кол-во доз АПГ в авт.реж.',
              'Напр. эл-ра', 'Шум', 'Пена: снято с эл-ра', 'Пустота анода', 'Уров. КПКВТ', 'Темп-ра КПКВТ']
    
    # Определение временных границ для обучения и тестирования
    start_date = pd.to_datetime('2019.04.21')  # Начальная дата обучающей выборки
    dif_date = pd.to_datetime('2021.04.21')    # Конечная дата обучающей выборки
    
    # Предобработка данных: очистка и форматирование
    df = arl_data.Data.fix_initial_frame(init_df, params, 'Кол-во трещин на аноде', 'Номер электролизера', 'Дата')
    index = pd.IndexSlice  # Инструмент для сложного индексирования в Pandas

    # Подготовка данных для 5-го корпуса
    # Обучающая выборка: данные от start_date до dif_date
    train_df = df.loc[index[objt_list_5, start_date:dif_date], :]
    # Тестовая выборка: данные после dif_date
    test_df = df.loc[index[objt_list_5, dif_date + pd.Timedelta(days=1):], :]

    print('Выполнено')
    print('Расчет модели')

    # Создание модели анализа формальных понятий с параметрами:
    # - тип бинаризации: по стандартному отклонению (STDDEV)
    # - индивидуальный тип: False (строится общая модель для всех электролизеров)
    # - keep_nan: False (не сохранять пропущенные значения)
    model = arl_fca_lab.arl_fca_lab(bin_type=arl_binarization.BinarizationType.STDDEV, ind_type=False, keep_nan=False)
    
    # Обучение модели на обучающей выборке
    model.fit_model(train_df, arl_data.Data.defect_name, arl_data.Data.cell_name, arl_data.Data.date_name, anomalies_only=False)

    print('Выполнено')
    print('Подготовка тестового датасета')

    # Преобразование тестовых данных с использованием полученной модели бинаризации
    test_matrix = model.bin_matrix.transform(test_df, model.defect, arl_data.Data.cell_name, arl_data.Data.date_name,
                                             model.keep_nan, model.days_before_defect, model.anomalies_only)

    print('Выполнено')
    print('Расчет оценок')

    # Создание DataFrame для хранения результатов тестирования
    test_results = pd.DataFrame(index=test_matrix.index, columns=['Object', 'Prediction', 'Fact', 'Rule'])

    # Итерация по всем строкам тестовой матрицы для получения предсказаний
    for i in test_matrix.index:
        # Определение номера объекта (если ind_type=True, то индивидуально, иначе 'all')
        if model.ind_type:
            obj_num = test_matrix.loc[i, arl_data.Data.cell_name]
        else:
            obj_num = 'all'
        
        # Получение предсказания для текущей строки
        prediction = model.get_prediction(arl_fca_lab.get_row(test_matrix, i), obj_num)
        
        # Запись информации в результирующий DataFrame
        test_results.loc[i, 'Object'] = test_matrix.loc[i, arl_data.Data.cell_name]
        test_results.loc[i, 'Fact'] = test_matrix.loc[i, '1_day_before']

        # Обработка случая, когда предсказание отсутствует
        if prediction == None:
            test_results.loc[i, 'Prediction'] = 0
            test_results.loc[i, 'Rule'] = -1
        else:
            test_results.loc[i, 'Prediction'] = prediction[0]  # Значение предсказания (confidence)
            test_results.loc[i, 'Rule'] = prediction[1]       # Номер правила

    print('Выполнено')
    
    # Добавление дополнительных столбцов с информацией из тестовой матрицы
    test_results = pd.concat([test_results, test_matrix[[arl_data.Data.date_name, arl_data.Data.defect_name, '1_day_before']]], axis=1)

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
            threshold_f1['Threshold'], threshold_f1['FP'] / len(test_results[(test_results[arl_data.Data.defect_name] == 0)&(test_results['Fact'] == 0)]), 'r')

    # Настройка отображения графика
    plt.xlabel("Пороговое значение")
    plt.ylabel("Доля")
    plt.legend(['Истинно положительные', 'Ложно положительные'])
    # Сохранение графика в файл
    plt.savefig('k5_std_I-F_N-F_A-F_s2019.png')
    plt.clf()  # Очистка текущей фигуры