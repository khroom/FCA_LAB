import pandas as pd  # для работы с табличными данными
import unidecode    # для транслитерации Unicode-символов в ASCII


class Data:
    # Класс для хранения и обработки данных
    df = None         # будет хранить основной DataFrame
    cell_name = None  # имя колонки с ячейками/клетками
    date_name = None  # имя колонки с датами
    defect_name = None  # имя колонки с дефектами

    @staticmethod
    def fix_initial_frame(df: pd.DataFrame, param_names: list, defect_name: str, 
                         cell_name: str, date_name: str, fullness=0.9):
        """
        Обрабатывает исходный DataFrame, оставляя только релевантные колонки и нормализуя имена
        
        Параметры:
        df - исходный DataFrame
        param_names - список параметров для анализа
        defect_name - имя колонки с дефектами
        cell_name - имя колонки с номерами электролизеров
        date_name - имя колонки с датами
        fullness - порог заполненности (0-1), по умолчанию 0.9 (90%)
        
        Возвращает:
        Обработанный DataFrame с мультииндексом (ячейка, дата)
        """
        
        # 1. Отбор колонок с достаточной заполненностью
        # Считаем количество непустых значений в указанных колонках
        counts = df[param_names].count()
        # Отбираем колонки, где заполнено >= fullness (90% по умолчанию) данных
        columns = counts[counts >= len(df) * fullness].index.to_list()
        
        # 2. Добавляем обязательные колонки (дефекты, ячейки, даты)
        columns.append(defect_name)
        columns.append(cell_name)
        columns.append(date_name)
        
        # 3. Создаем новый DataFrame только с отобранными колонками
        # reset_index() сохраняет старый индекс как колонку
        result = df.reset_index()[columns]
        
        # 4. Нормализация имен колонок:
        # - удаляем акценты и спецсимволы (unidecode)
        # - удаляем апострофы
        for column in result.columns:
            result.rename(
                columns={column: unidecode.unidecode(column).replace('\'', '')}, 
                inplace=True
            )
        
        # 5. Сохраняем обработанные имена ключевых колонок в атрибуты класса
        Data.cell_name = unidecode.unidecode(cell_name).replace('\'', '')
        Data.date_name = unidecode.unidecode(date_name).replace('\'', '')
        Data.defect_name = unidecode.unidecode(defect_name).replace('\'', '')
        
        # 6. Устанавливаем мультииндекс (ячейка + дата) и сортируем
        Data.df = result.set_index([Data.cell_name, Data.date_name]).sort_index()
        
        return Data.df
