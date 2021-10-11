import arl_fca_lab
import arl_binarization
import arl_data
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('.\\kras_binary\\resultall KRAZ.csv', parse_dates=['Дата'])
    params = ['Эл-лит: КО', 'Эл-лит: CaF2', 'Ток серии (КПП)', 'Металл: уровень',
        'Эл-т: уровень', 'Эл-лит: темп-ра', 'Металл: Fe', 'Металл: Si',
       'Выливка: Л/О', 'Время на голод.', 'РМПР: кол-во ВИРА',
       'РМПР: длит. ВИРА', 'РМПР: кол-во МАЙНА', 'РМПР: длит. МАЙНА',
       'РМПР: коэфф.', 'Кол-во доз АПГ в авт.реж.', 'Время АПГ в руч.реж.',
       'Время запрета АПГ', 'Напр. эл-ра', 'Напр. зад.', 'Доб. напр.', 'Шум',
       'Длит. шума', 'Напр. анода', 'Обр. ЭДС', 'AlF3: добавка',
       'Вторичный криолит: добавка', 'Пена: снято с эл-ра', 'Выливка: Кр/весы',
       'Выливка: задание', 'Уставка АПГ', 'Состояние КПК',
       'Расст. до колокола', 'Пустота анода', 'Ножка', 'Уров. КПКВТ',
       'Темп-ра КПКВТ', 'Угольная пена', 'ПАК: время запрета', 'ПАК: кол-во',
       'ПАК: сумм. длит. авт.', 'ПАК: сумм. длит. руч.',
       'Обработанная сторона', 'Кол-во перетяжек ан.рамы',
        'Срок службы', 'Эл-лит: MgF2', 'Напр. прив.',
        'Период шума', 'Напряжение ошиновки', 'Кол-во доз АПГ в руч.реж.', 'AlF3: кол-во доз в авт.реж.']

    faults = ['Кол-во нарушений на аноде','Кол-во трещин на аноде','Кол-во кусков под анодом', 'АЭ: кол-во', 'АЭ: напр. ср.', 'АЭ: длит.']

    df2 = arl_data.Data.fix_initial_frame(df, params, faults, 'Номер электролизера','Дата')
    # Исправление ошибки в номере электролизера было 50000, стало 5000, для всех подобных случаев
    # df['Номер электролизера'] = pd.Series([i if i < 50000 else i - 45000 for i in df['Номер электролизера']])
    # objt_list_5 = [i for i in df['Номер электролизера'].unique() if i < 6000]
    # objt_list_6 = [i for i in df['Номер электролизера'].unique() if i >= 6000]
