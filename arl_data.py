import pandas as pd
import unidecode


class Data:
    
    @staticmethod
    def fix_initial_frame(df: pd.DataFrame, param_names: list, faults_names: list, cell_name: str, date_name: str, fullness=0.9):
        counts = df[param_names].count()
        columns = counts[counts >= len(df) * fullness].index.to_list()
        columns = columns + faults_names
        columns.append(cell_name)
        columns.append(date_name)
        result = df.reset_index()[columns]
        for column in result.columns:
            result.rename(columns={column: unidecode.unidecode(column).replace('\'', '')}, inplace=True)
        uni_cell_name = unidecode.unidecode(cell_name).replace('\'', '')
        uni_date_name = unidecode.unidecode(date_name).replace('\'', '')
        return result.set_index([uni_cell_name, uni_date_name])
