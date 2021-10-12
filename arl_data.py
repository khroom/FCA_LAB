import pandas as pd
import unidecode


class Data:
    df = None
    cell_name = None
    date_name = None
    defect_name = None

    @staticmethod
    def fix_initial_frame(df: pd.DataFrame, param_names: list, defect_name: str, cell_name: str, date_name: str, fullness=0.9):
        counts = df[param_names].count()
        columns = counts[counts >= len(df) * fullness].index.to_list()
        columns.append(defect_name)
        columns.append(cell_name)
        columns.append(date_name)
        result = df.reset_index()[columns]
        for column in result.columns:
            result.rename(columns={column: unidecode.unidecode(column).replace('\'', '')}, inplace=True)
        Data.cell_name = unidecode.unidecode(cell_name).replace('\'', '')
        Data.date_name = unidecode.unidecode(date_name).replace('\'', '')
        Data.defect_name = unidecode.unidecode(defect_name).replace('\'', '')
        Data.df = result.set_index([Data.cell_name, Data.date_name]).sort_index()
        return Data.df
