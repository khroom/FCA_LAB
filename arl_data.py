import pandas as pd
import unidecode


class Data:
    
    @staticmethod
    def fix_initial_frame(df: pd.DataFrame, fullness=0.9):
        service_cols = ['Индекс', 'АЭ: кол-во', 'АЭ: длит.', 'АЭ: напр. ср.', 'Пена: снято с эл-ра', 'Срок службы',
                        'Кол-во поддер. анодов', 'АЭ: кол-во локальных', 'Отставание (шт)', 'Другое нарушение (шт)',
                        'Нарушение', 'Анод', 'Выливка: отношение скорости', 'Выход по току: Л/О', 'Подина: напр.',
                        'Кол-во доз АПГ в руч.реж.', 'Конус (да/нет)']
        result_frame = df.drop(service_cols, axis='columns')
        not_full_columns = []
        for column in result_frame.columns:
            if result_frame[column].count() / len(result_frame) < fullness:
                not_full_columns.append(column)
        result_frame = result_frame.drop(not_full_columns, axis='columns')
        for column in result_frame.columns:
            result_frame.rename(columns={column: unidecode.unidecode(column).replace('\'', '')}, inplace=True)
        result_frame = result_frame.set_index(['Nomer elektrolizera', 'Data'])
        return result_frame
