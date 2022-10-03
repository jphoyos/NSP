import numpy as np
import scipy as sp
import pandas as pd
# import matplotlib.pyplot as plt

import os
import re

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Consolidator:
    def __init__(self,raw_data_folder,consolidated_data_location,citas_excel_location=None):
        ## Abrir y leer archivos Excel
        self.consolidated_data_location = consolidated_data_location
        self.directorio = raw_data_folder

        # Definir dataframe
        self.Datos = pd.DataFrame([])

        # Expresion regular que busca archivos con nombre INFORME ESTADISTICO
        self.newlist_Informe = []
        for filename in os.listdir(self.directorio):
            if (filename.endswith("xlsx") & filename.startswith("~$") == False):
                self.newlist_Informe.append(self.directorio+filename)
        if citas_excel_location != None:
            self.citas_excel_location = citas_excel_location
        else:
            self.citas_excel_location = None
        logger.info( 'Los informes excel son {}'.format(self.newlist_Informe))

    def consolidate(self,idx="PAID"):
        # El archivo Base.csv sera donde se guarde la base de datos completa. 
        # Si no esta en el directorio, se crea
        
        
        columns = [idx,u'FechaNac', u'Sexo', u'Comuna', u'Prevision',
                  u'Especialidad', u'TipoAtencion', 'TipoProfesional', 'CodPrestacion',
                  u'FechaCita', u'EstadoCita', u'HoraCita']
        if self.citas_excel_location != None:
            self.citas_datos = pd.read_excel(self.citas_excel_location, parse_dates=['FechaNac','FechaCita','FechaReserva'])[columns[:-1] + ['FechaReserva']]
            self.citas_datos[idx] = pd.to_numeric(self.citas_datos[idx],errors="coerce",downcast="integer")
            logger.info("citas_datos shape: {}".format(self.citas_datos.shape))
            self.citas_datos.dropna(inplace=True)
            logger.info("citas_datos shape: {}".format(self.citas_datos.shape))
            self.use_citas_datos = True
        else:
            self.citas_datos = None
            self.use_citas_datos = False
        logger.info('Leer todos los excels para compilarlos')
        for archivo in self.newlist_Informe:

            # Abro el archivo excel
            df = pd.ExcelFile(archivo)
            # Leo la primera hoja del excel
            try:
                DatosDF = df.parse(sheet_name=df.sheet_names[0],parse_dates=['FechaNac','FechaCita'])
                # DatosDF[idx] = pd.to_numeric(DatosDF[idx],errors="coerce",downcast="integer")
            
                DatosDF = DatosDF[columns]
                self.Datos = pd.concat([self.Datos,DatosDF])
                logger.info("{} cargado".format(archivo))
            except KeyError as e:
                logger.info("{} error: {}".format(archivo,e))
            except ValueError as e:
                logger.info("{} error: {}".format(archivo,e))
        logger.info("Datos shape: {}".format(self.Datos.shape))
        self.Datos.dropna(inplace=True)
        logger.info("Datos shape: {}".format(self.Datos.shape))
        if self.use_citas_datos:
            self.Datos = self.Datos.merge(self.citas_datos[['FechaNac','FechaCita','CodPrestacion','FechaReserva']],how="left")
        logger.info("Datos shape: {}".format(self.Datos.shape))
        logger.info(self.Datos.columns)

        self.Datos.dropna(inplace=True)
        logger.info("Datos shape: {}".format(self.Datos.shape))
        
        #Datos.to_csv('Base.csv', sep='\t', encoding='utf-8',index=True)
        self.Datos.to_csv(self.consolidated_data_location, encoding='utf-8',index=False)

class ConsolidatorCrsco:
    def __init__(self,raw_data_folder):
        self.raw_files=[]
        for filename in os.listdir(raw_data_folder):
            if filename.endswith(".xlsx") and not filename.startswith("~$"):
                self.raw_files.append(raw_data_folder+filename)
        logger.info("files to consolidate: " + str(self.raw_files))
    def consolidate(self,consolidated_filepath):
        columns = ["Rut",u'FechadeNac', u'Sexo', u'Comuna', u'Prevision',
                  u'Especialidad', u'TipoAtencion', 'TipoProfesional', 'CodigoPrestacion',
                  u'FechaCita', u'EstadoCita', u'HoraCita']
        dfs = []
        for filename in self.raw_files:
            logger.info(filename + " loading")
            df = pd.read_excel(filename, parse_dates=['FechadeNac','FechaCita','HoraCita'])[columns]
            dfs.append(df)
            logger.info(df.shape)
            df.dropna(inplace=True)
            logger.info(df.shape)
            logger.info(filename + " loaded")
        self.data = pd.concat(dfs, axis=0, ignore_index=True)
        self.data.to_csv(consolidated_filepath, index=False)

class ConsolidatorHrt:
    def __init__(self,raw_data_folder):
        self.raw_files=[]
        for filename in os.listdir(raw_data_folder):
            if filename.endswith(".xlsx") and not filename.startswith("~$"):
                self.raw_files.append(raw_data_folder+filename)
        logger.info("files to consolidate: " + str(self.raw_files))
    def consolidate(self,consolidated_filepath):
        columns = ["RUT",u'FECHANAC', u'SEXO', u'COMUNA', u'FONASA',
                  u'ESPECIALIDAD', u'TIPO_ATENCION', 'TIPO_PROFESIONAL',
                  u'FECHA_CITA', u'HORA_CITA', 'FECHA_RESERVA',u'FECHA_CONFIRMACION']
        dfs = []
        for filename in self.raw_files:
            logger.info(filename + " loading")
            data = pd.read_excel(filename, parse_dates=['FECHANAC','FECHA_CITA','HORA_CITA','FECHA_RESERVA'],sheet_name=None)
            for sheet,df in data.items():
                df = df[columns]
                logger.info(df.shape)
                df["ESPECIALIDAD"] = df.ESPECIALIDAD.str.extract(r'(.*) - ',expand=False)
                df.dropna(inplace=True,subset=columns[:-1])
                dfs.append(df)
                logger.info(df.shape)
                logger.info(filename + " sheet " + sheet + " loaded")
        self.data = pd.concat(dfs, axis=0, ignore_index=True)
        self.data.to_csv(consolidated_filepath, index=False)