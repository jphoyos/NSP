import pandas as pd
import numpy as np
from sqlalchemy import create_engine

class HistoryCreator:
    def __init__(self,dataset,columns=["PAID","FechaCita","Especialidad"],label_column="EstadoCita",true_nsp="No Atendido",false_nsp="Atendido"):
        self.data = pd.read_csv(dataset)[columns+[label_column]]
        self.true_nsp = true_nsp
        self.false_nsp = false_nsp
    def create_history(self):
        self.data = self.data[self.data["EstadoCita"].isin([self.false_nsp,self.true_nsp])]
        self.data["NSP"] = np.where(self.data["EstadoCita"] == self.true_nsp,1,0)
        self.data.drop(columns=["EstadoCita"], inplace=True, axis=1)
        self.data["FechaCita"] = pd.to_datetime(self.data["FechaCita"]).dt.date
        self.data.dropna(inplace=True)
        self.data.drop_duplicates(inplace=True)
    def write_db(self,db_location):
        engine = create_engine(r'sqlite:///'+db_location, echo=True)
        self.data.to_sql('history', con=engine,if_exists="replace",index=False)

class HistoryCreatorHrt:
    def __init__(self,dataset,columns=["RUT","FECHA_CITA","ESPECIALIDAD"],label_column="FECHA_CONFIRMACION"):
        self.data = pd.read_csv(dataset)[columns+[label_column]]
        self.label_column = label_column
        self.columns = columns
    def create_history(self):
        self.data["NSP"] = np.where(self.data[self.label_column].isna(),1,0)
        self.data.drop(columns=[self.label_column], inplace=True, axis=1)
        self.data[self.columns[1]] = pd.to_datetime(self.data[self.columns[1]]).dt.date
        self.data.dropna(inplace=True)
        self.data.drop_duplicates(inplace=True)
    def write_db(self,db_location):
        engine = create_engine(r'sqlite:///'+db_location, echo=True)
        self.data.to_sql('history', con=engine,if_exists="replace",index=False)
