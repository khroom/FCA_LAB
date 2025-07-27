# Импорт необходимых библиотек
import arl_fca_lab       # Библиотека для анализа формальных понятий (FCA)
import arl_binarization  # Библиотека для бинаризации данных
import arl_data          # Вспомогательные функции для работы с данными
import pandas as pd      # Для работы с табличными данными
import numpy as np       # Для численных операций
import matplotlib.pyplot as plt  # Для визуализации данных


# Основной блок выполнения программы
if __name__ == '__main__':
    print('Загрузка исходных данных')
    # Загрузка данных из CSV-файла с автоматическим парсингом даты в столбце 'Дата'
    # Файл находится в поддиректории kras_binary
    init_df = pd.read_csv('.\\kras_binary\\result.csv', parse_dates=['Дата'])

    # Коррекция номеров электролизеров: 
    # Если номер >= 50000, вычитаем 45000 (например, 50000 становится 5000)
    # Это исправление ошибки в исходных данных
    init_df['Номер электролизера'] = pd.Series(
        [i if i < 50000 else i - 45000 for i in init_df['Номер электролизера']]
    )

    # Разделение электролизеров на две группы:
    # - 5-й корпус: номера < 6000
    # - 6-й корпус: номера >= 6000
    objt_list_5 = [i for i in init_df['Номер электролизера'].unique() if i < 6000]
    objt_list_6 = [i for i in init_df['Номер электролизера'].unique() if i >= 6000]

    # Список параметров, которые будут анализироваться
    params = [
        'Срок службы', 'Эл-лит: темп-ра', 'Выливка: Л/О', 'РМПР: коэфф.', 
        'Кол-во доз АПГ в авт.реж.', 'Напр. эл-ра', 'Шум', 'Пена: снято с эл-ра', 
        'Пустота анода', 'Уров. КПКВТ', 'Темп-ра КПКВТ'
    ]
    
    # Установка временных границ:
    start_date = pd.to_datetime('2019.04.21')  # Начало обучающего периода
    dif_date = pd.to_datetime('2021.04.21')     # Конец обучающего периода
    
    # Предобработка данных:
    # - Очистка данных
    # - Выбор указанных параметров
    # - Установка целевой переменной ('Кол-во трещин на аноде')
    df = arl_data.Data.fix_initial_frame(
        init_df, params, 'Кол-во трещин на аноде', 'Номер электролизера', 'Дата'
    )
    
    # Инструмент для сложного индексирования в Pandas
    index = pd.IndexSlice

    # Подготовка данных для 5-го корпуса:
    # Обучающая выборка - данные от start_date до dif_date
    train_df = df.loc[index[objt_list_5, start_date:dif_date], :]
    # Тестовая выборка - данные после dif_date
    test_df = df.loc[index[objt_list_5, dif_date + pd.Timedelta(days=1):], :]

    print('Выполнено')
    print('Расчет модели')

    # Создание модели анализа формальных понятий с параметрами:
    # - bin_type=STDDEV: бинаризация по стандартному отклонению
    # - ind_type=False: не использовать индивидуальные настройки для каждого объекта
    # - keep_nan=False: не сохранять пропущенные значения
    model = arl_fca_lab.arl_fca_lab(
        bin_type=arl_binarization.BinarizationType.STDDEV, 
        ind_type=False, 
        keep_nan=False
    )
    
    # Обучение модели на обучающей выборке:
    # - train_df: обучающие данные
    # - defect_name: имя столбца с целевой переменной
    # - cell_name: имя столбца с номерами электролизеров
    # - date_name: имя столбца с датами
    # - anomalies_only=False: использовать все данные, не только аномалии
    model.fit_model(
        train_df, 
        arl_data.Data.defect_name, 
        arl_data.Data.cell_name, 
        arl_data.Data.date_name, 
        anomalies_only=False
    )

    print('Выполнено')
    print('Подготовка тестового датасета')

    # Преобразование тестовых данных с использованием обученной модели:
    # - Бинаризация данных
    # - Учет настроек модели (keep_nan, days_before_defect, anomalies_only)
    test_matrix = model.bin_matrix.transform(
        test_df, 
        model.defect, 
        arl_data.Data.cell_name, 
        arl_data.Data.date_name,
        model.keep_nan, 
        model.days_before_defect, 
        model.anomalies_only
    )

    print('Выполнено')
    print('Расчет оценок')

    # Создание DataFrame для хранения результатов:
    # - Object: номер электролизера
    # - Prediction: предсказанное значение
    # - Fact: фактическое значение
    # - Rule: номер правила, по которому сделано предсказание
    test_results = pd.DataFrame(
        index=test_matrix.index, 
        columns=['Object', 'Prediction', 'Fact', 'Rule']
    )

    # Итерация по всем строкам тестовой выборки
    for i in test_matrix.index:
        # Определение номера объекта для предсказания
        if model.ind_type:
            obj_num = test_matrix.loc[i, arl_data.Data.cell_name]  # Индивидуальный номер
        else:
            obj_num = 'all'  # Общая модель для всех объектов
        
        # Получение предсказания для текущей строки
        prediction = model.get_prediction(arl_fca_lab.get_row(test_matrix, i), obj_num)
        
        # Запись информации об объекте и фактическом значении
        test_results.loc[i, 'Object'] = test_matrix.loc[i, arl_data.Data.cell_name]
        test_results.loc[i, 'Fact'] = test_matrix.loc[i, '1_day_before']

        # Обработка результатов предсказания
        if prediction is None:
            # Если предсказание отсутствует
            test_results.loc[i, 'Prediction'] = 0  # Значение по умолчанию
            test_results.loc[i, 'Rule'] = -1      # Специальный код для отсутствующего правила
        else:
            # Запись предсказания и номера правила
            test_results.loc[i, 'Prediction'] = prediction[0]  # Вероятность/оценка (confidence)
            test_results.loc[i, 'Rule'] = prediction[1]       # Номер правила

    print('Выполнено')
    
    # Добавление дополнительных столбцов из тестовой матрицы:
    # - Дата
    # - Значение дефекта
    # - Значение за день до события
    test_results = pd.concat([
        test_results, 
        test_matrix[[
            arl_data.Data.date_name, 
            arl_data.Data.defect_name, 
            '1_day_before'
        ]]
    ], axis=1)

    # Расчет метрик качества для разных пороговых значений (от 0.01 до 0.99 с шагом 0.01)
    threshold_f1 = pd.DataFrame()
    for threshold in np.arange(0.01, 0.99, 0.01):
        # True Positive (TP): предсказали дефект, когда он действительно был
        threshold_f1.loc[threshold, 'TP'] = len(
            test_results[(test_results.Fact == 1) & (test_results.Prediction >= threshold)]
        )
        
        # False Positive (FP): предсказали дефект, когда его не было
        threshold_f1.loc[threshold, 'FP'] = len(
            test_results[(test_results.Fact == 0) & (test_results.Prediction >= threshold)]
        )
        
        # False Negative (FN): не предсказали дефект, когда он был
        threshold_f1.loc[threshold, 'FN'] = len(
            test_results[(test_results.Fact == 1) & (test_results.Prediction < threshold)]
        )
        
        # True Negative (TN): правильно предсказали отсутствие дефекта
        threshold_f1.loc[threshold, 'TN'] = len(
            test_results[(test_results.Fact == 0) & (test_results.Prediction < threshold)]
        )
        
        # Сумма правильных предсказаний (TN + TP)
        threshold_f1.loc[threshold, 'TN_TP'] = (
            threshold_f1.loc[threshold, 'TN'] + threshold_f1.loc[threshold, 'TP']
        )
        
        # Сохранение порогового значения
        threshold_f1.loc[threshold, 'Threshold'] = threshold

    # Визуализация результатов
    fig, ax = plt.subplots()  # Создание фигуры и осей
    
    # Построение графиков:
    # Зеленый: доля истинно положительных результатов (TP-частота)
    # Красный: доля ложно положительных результатов (FP-частота)
    ax.plot(
        threshold_f1['Threshold'], 
        threshold_f1['TP'] / len(test_results[test_results['Fact'] == 1]), 
        'g',
        threshold_f1['Threshold'], 
        threshold_f1['FP'] / len(test_results[
            (test_results[arl_data.Data.defect_name] == 0) & 
            (test_results['Fact'] == 0)
        ]), 
        'r'
    )

    # Настройка отображения графиков
    plt.xlabel("Пороговое значение")  # Подпись оси X
    plt.ylabel("Доля")                # Подпись оси Y
    plt.legend([
        'Истинно положительные',  # Легенда для зеленой линии
        'Ложно положительные'     # Легенда для красной линии
    ])
    
    # Сохранение графика в файл
    plt.savefig('k5_std_I-F_N-F_A-F_s2019.png')
    plt.clf()  # Очистка текущей фигуры