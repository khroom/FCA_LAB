import arl_fca_lab
import arl_binarization
import arl_data
import pandas as pd
import numpy
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print('Загрузка исходных данных')
    init_df = pd.read_csv('.\\kras_binary\\result.csv', parse_dates=['Дата'])

    # Исправление ошибки в номере электролизера было 50000, стало 5000, для всех подобных случаев
    init_df['Номер электролизера'] = pd.Series([i if i < 50000 else i - 45000 for i in init_df['Номер электролизера']])

    objt_list_5 = [i for i in init_df['Номер электролизера'].unique() if i < 6000]
    objt_list_6 = [i for i in init_df['Номер электролизера'].unique() if i >= 6000]

    params = ['Срок службы', 'Эл-лит: темп-ра', 'Выливка: Л/О', 'РМПР: коэфф.', 'Кол-во доз АПГ в авт.реж.',
              'Напр. эл-ра', 'Шум', 'Пена: снято с эл-ра', 'Пустота анода', 'Уров. КПКВТ', 'Темп-ра КПКВТ']
    start_date = pd.to_datetime('2019.04.21')
    dif_date = pd.to_datetime('2021.04.21')
    df = arl_data.Data.fix_initial_frame(init_df, params, 'Кол-во трещин на аноде', 'Номер электролизера', 'Дата')
    index = pd.IndexSlice

    # Исследование для пятого корпуса
    train_df = df.loc[index[objt_list_5, start_date:dif_date], :]
    test_df = df.loc[index[objt_list_5, dif_date + pd.Timedelta(days=1):], :]

    print('Выполнено')
    print('Расчет модели')

    model = arl_fca_lab.arl_fca_lab(bin_type=arl_binarization.BinarizationType.STDDEV, ind_type=False, keep_nan=False)
    model.fit_model(train_df, arl_data.Data.defect_name, arl_data.Data.cell_name, arl_data.Data.date_name, anomalies_only=False)

    print('Выполнено')
    print('Подготовка тестового датасета')

    test_matrix = model.bin_matrix.transform(test_df, model.defect, arl_data.Data.cell_name, arl_data.Data.date_name,
                                             model.keep_nan, model.days_before_defect, model.anomalies_only)

    print('Выполнено')
    print('Расчет оценок')

    test_results = pd.DataFrame(index=test_matrix.index, columns=['Object', 'Prediction', 'Fact', 'Rule'])

    for i in test_matrix.index:

        if model.ind_type:
            obj_num = test_matrix.loc[i, arl_data.Data.cell_name]
        else:
            obj_num = 'all'
        prediction = model.get_prediction(arl_fca_lab.get_row(test_matrix, i), obj_num)
        test_results.loc[i, 'Object'] = test_matrix.loc[i, arl_data.Data.cell_name]
        test_results.loc[i, 'Fact'] = test_matrix.loc[i, '1_day_before']

        if prediction == None:
            test_results.loc[i, 'Prediction'] = 0
            test_results.loc[i, 'Rule'] = -1
        else:
            test_results.loc[i, 'Prediction'] = prediction[0]
            test_results.loc[i, 'Rule'] = prediction[1]
            # model.confidence_df[obj_num].loc[rule_index, 'B']
        # print('Index:', i, ', Object:', test_results.loc[i, 'Object'], ', Predict:', test_results.loc[i, 'Prediction'],
        #       ', Fact:', test_results.loc[i, 'Fact'])

    print('Выполнено')
    test_results = pd.concat([test_results, test_matrix[[arl_data.Data.date_name, arl_data.Data.defect_name, '1_day_before']]], axis=1)

    threshold_f1= pd.DataFrame()
    for threshold in numpy.arange(0.01, 0.99, 0.01):
        threshold_f1.loc[threshold, 'TP'] = len(
            test_results[(test_results.Fact == 1)&(test_results.Prediction >= threshold)])
        threshold_f1.loc[threshold, 'FP'] = len(
            test_results[(test_results.Fact == 0)&(test_results.Prediction >= threshold)])
        threshold_f1.loc[threshold, 'FN'] = len(
            test_results[(test_results.Fact == 1) & (test_results.Prediction < threshold)])
        threshold_f1.loc[threshold, 'TN'] = len(
            test_results[(test_results.Fact == 0) & (test_results.Prediction < threshold)])
        threshold_f1.loc[threshold, 'TN_TP'] = threshold_f1.loc[threshold, 'TN']+threshold_f1.loc[threshold, 'TP']
        threshold_f1.loc[threshold, 'Threshold'] = threshold

    fig, ax = plt.subplots()
    ax.plot(threshold_f1['Threshold'], threshold_f1['TP'] / len(test_results[test_results['Fact'] == 1]), 'g',
            threshold_f1['Threshold'], threshold_f1['FP'] / len(test_results[(test_results[arl_data.Data.defect_name] == 0)&(test_results['Fact'] == 0)]), 'r')

    # plt.legend(['Истинно положительных', 'Ложно положительных'])
    plt.xlabel("Пороговое значение")
    plt.ylabel("Доля")
    # plt.title('Vanna-9001, bin_type=STDDEV, ind_type=True, \nkeep_nan=True, anomalies_only=False, \n'
    #           'train(2018.01.01-2019.01.01), test(2019.01.01-2019.08.01)')
    plt.legend(['Истинно положительные', 'Ложно положительные'])
    plt.savefig('k5_std_I-F_N-F_A-F_s2019.png')
    plt.clf()
    # plt.show()

