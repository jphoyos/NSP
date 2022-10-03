#!/usr/bin/env python

# coding: utf-8

import pandas as pd

import numpy as np

import hashlib


def get_data(
        hrt_db_url, hlcm_db_url, crsco_db_url, crsco_state_url):

    # HRT

    hrt = pd.read_excel(hrt_db_url, sheet_name=None)

    hrt_data = hrt["raw"].copy()

    hrt_data["predicted_as_nsp"] = hrt_data["DAC_CORRELATIVO"].isin(
        hrt["predicted"]["DAC_CORRELATIVO"])

    hrt_data = hrt_data.merge(hrt["nsp"], how="left")

    hrt_data["group"] = np.nan

    hrt_data["group"] = np.where(hrt_data["DAC_CORRELATIVO"].isin(
        hrt["ge1"]["DAC_CORRELATIVO"]), "phone_call", hrt_data["group"])

    hrt_data["group"] = np.where(hrt_data["DAC_CORRELATIVO"].isin(
        hrt["ge2"]["DAC_CORRELATIVO"]), "sms_message", hrt_data["group"])

    hrt_data["group"] = np.where(hrt_data["DAC_CORRELATIVO"].isin(
        hrt["ge3"]["DAC_CORRELATIVO"]), "whatsapp_message", hrt_data["group"])

    hrt_data["group"] = np.where(hrt_data["DAC_CORRELATIVO"].isin(
        hrt["gc"]["DAC_CORRELATIVO"]), "control", hrt_data["group"])

    hrt_data.replace("nan", np.nan, inplace=True)

    hrt_data = hrt_data.merge(hrt["ge1"][[
                              "DAC_CORRELATIVO", "attemp", "comentario"]], how="left", on="DAC_CORRELATIVO")

    hrt_data = hrt_data.merge(
        hrt["ge2"][["DAC_CORRELATIVO", "sent"]], how="left", on="DAC_CORRELATIVO")

    hrt_data = hrt_data.merge(
        hrt["ge3"][["DAC_CORRELATIVO", "sent"]], how="left", on="DAC_CORRELATIVO")

    hrt_data["sent"] = np.where(
        hrt_data.sent_x.isna(), hrt_data.sent_y, hrt_data.sent_x)

    hrt_data["RUT"] = hrt_data.RUT.astype(str) + "-" + hrt_data.DV.astype(str)

    hrt_data.drop(['sent_x', 'sent_y', "DV", "USUARIO_CONFIRMA",
                  "AGENDADO_POR"], axis=1, inplace=True)

    hrt_data.rename(columns={
                    "attemp": "phone_call_answer_attemp", "sent": "message_sent"}, inplace=True)

    hrt_data = hrt_data.merge(hrt["qualitative"][["DAC_CORRELATIVO", "record", "action_received", "time",
                              "helped", "obs", "no_show_reason", "liked_method", "answered"]], how="left", on="DAC_CORRELATIVO")

    hrt_data.rename(
        columns={"answered": "qualitative_interview_answered"}, inplace=True)

    # HLCM

    hlcm = pd.read_excel(hlcm_db_url, sheet_name=None)

    hlcm_data = hlcm["data"].copy()

    hlcm_data["predicted_as_nsp"] = hlcm_data["APPT_RowId"].isin(
        hlcm["predicted"]["APPT_RowId"])

    hlcm_data = hlcm_data.merge(
        hlcm["predicted"][["APPT_RowId", "EstadoCita"]], on="APPT_RowId", how="left")

    hlcm_data["nsp"] = np.where(hlcm_data.EstadoCita_y == "Atendido", 0, np.where(
        hlcm_data.EstadoCita_y == "No atendido", 1, np.nan))

    hlcm_data.drop(["EstadoCita_x"], axis=1, inplace=True)

    hlcm_data.rename(columns={"EstadoCita_y": "EstadoCita"}, inplace=True)

    hlcm_data["group"] = np.nan

    hlcm_data["group"] = np.where(hlcm_data["APPT_RowId"].isin(
        hlcm["ge1"]["APPT_RowId"]), "phone_call", hlcm_data["group"])

    hlcm_data["group"] = np.where(hlcm_data["APPT_RowId"].isin(
        hlcm["ge2"]["APPT_RowId"]), "sms_message", hlcm_data["group"])

    hlcm_data["group"] = np.where(hlcm_data["APPT_RowId"].isin(
        hlcm["ge3"]["APPT_RowId"]), "whatsapp_message", hlcm_data["group"])

    hlcm_data["group"] = np.where(hlcm_data["APPT_RowId"].isin(
        hlcm["gc"]["APPT_RowId"]), "control", hlcm_data["group"])

    hlcm_data.replace("nan", np.nan, inplace=True)

    hlcm_data = hlcm_data.merge(
        hlcm["ge1"][["APPT_RowId", "Intentos", "Comentario"]], how="left", on="APPT_RowId")

    hlcm_data = hlcm_data.merge(
        hlcm["ge2"][["APPT_RowId", "sent"]], how="left", on="APPT_RowId")

    hlcm_data = hlcm_data.merge(
        hlcm["ge3"][["APPT_RowId", "sent"]], how="left", on="APPT_RowId")

    hlcm_data["APELLIDOS_PACIENTE"] = hlcm_data.PacienteApPaterno + \
        " " + hlcm_data.PacienteApMaterno

    hlcm_data["sent"] = np.where(
        hlcm_data.sent_x.isna(), hlcm_data.sent_y, hlcm_data.sent_x)

    hlcm_data.drop(['sent_x', 'sent_y', "PacienteApPaterno",
                   "PacienteApMaterno"], axis=1, inplace=True)

    hlcm_data.rename(columns={"Intentos": "phone_call_answer_attemp",
                     "sent": "message_sent", "Comentario": "comentario"}, inplace=True)

    hlcm_qualitative = hlcm["qualitative"]

    hlcm_qualitative = pd.concat([hlcm_qualitative, hlcm["qualitative_gc"]])

    hlcm_data = hlcm_data.merge(hlcm_qualitative[["APPT_RowId", "record", "action_received", "time",
                                "helped", "obs", "motivo_no_show", "liked_method", "Answered"]], how="left", on="APPT_RowId")

    hlcm_data.rename(columns={"motivo_no_show": "no_show_reason",
                     "Answered": "qualitative_interview_answered"}, inplace=True)

    # CRSCO

    crsco = pd.read_excel(
        crsco_db_url, sheet_name=None)

    crsco["data"].set_index(["Rut", "FechaCita", "HoraCita"],
                            verify_integrity=True, inplace=True, drop=False)

    crsco["predicted"].set_index(
        ["Rut", "FechaCita", "HoraCita"], verify_integrity=True, inplace=True, drop=False)

    crsco["ge1"].set_index(["Rut", "FechaCita", "HoraCita"],
                           verify_integrity=True, inplace=True, drop=False)

    crsco["ge2"].set_index(["Rut", "FechaCita", "HoraCita"],
                           verify_integrity=True, inplace=True, drop=False)

    crsco["ge3"].set_index(["Rut", "FechaCita", "HoraCita"],
                           verify_integrity=True, inplace=True, drop=False)

    crsco["gc"].set_index(["Rut", "FechaCita", "HoraCita"],
                          verify_integrity=True, inplace=True, drop=False)

    crsco["qualitative"].set_index(
        ["Rut", "FechaCita", "HoraCita"], verify_integrity=True, inplace=True, drop=False)

    crsco["qualitative_gc"].set_index(
        ["Rut", "FechaCita", "HoraCita"], verify_integrity=True, inplace=True, drop=False)

    crsco_data = crsco["data"].copy()

    crsco_data.drop("Unnamed: 47", axis=1, inplace=True)

    crsco_data["predicted_as_nsp"] = crsco_data.index.isin(
        crsco["predicted"].index)

    crsco_state = pd.read_excel(
        crsco_state_url, sheet_name=None)

    crsco_state["EstadoCitasGI"]["HoraCita"] = crsco_state["EstadoCitasGI"].HoraCita.astype(
        str).apply(pd.Timestamp).dt.time

    crsco_state["EstadoCitasGC"]["HoraCita"] = crsco_state["EstadoCitasGC"].HoraCita.astype(
        str).apply(pd.Timestamp).dt.time

    crsco_state = pd.concat(
        [crsco_state["EstadoCitasGI"], crsco_state["EstadoCitasGC"]])

    crsco_state.drop_duplicates(
        subset=["Rut", "FechaCita", "HoraCita"], inplace=True)

    crsco_state.set_index(["Rut", "FechaCita", "HoraCita"],
                          verify_integrity=True, inplace=True, drop=False)

    crsco_data = crsco_data.merge(
        crsco_state.EstadoCita, left_index=True, right_index=True, how="left")

    crsco_data["nsp"] = np.where(crsco_data.EstadoCita_y == "Atendido", 0, np.where(
        crsco_data.EstadoCita_y == "No Atendido", 1, np.nan))

    crsco_data.drop(["EstadoCita_x"], axis=1, inplace=True)

    crsco_data.rename(columns={"EstadoCita_y": "EstadoCita"}, inplace=True)

    crsco_data["group"] = np.nan

    crsco_data["group"] = np.where(crsco_data.index.isin(
        crsco["ge1"].index), "phone_call", crsco_data["group"])

    crsco_data["group"] = np.where(crsco_data.index.isin(
        crsco["ge2"].index), "sms_message", crsco_data["group"])

    crsco_data["group"] = np.where(crsco_data.index.isin(
        crsco["ge3"].index), "whatsapp_message", crsco_data["group"])

    crsco_data["group"] = np.where(crsco_data.index.isin(
        crsco["gc"].index), "control", crsco_data["group"])

    crsco_data.replace("nan", np.nan, inplace=True)

    crsco_data = crsco_data.merge(
        crsco["ge1"][["Attempt", "Comentario"]], how="left", left_index=True, right_index=True)

    crsco_data = crsco_data.merge(
        crsco["ge2"]["sent"], how="left", left_index=True, right_index=True)

    crsco_data = crsco_data.merge(
        crsco["ge3"]["sent"], how="left", left_index=True, right_index=True)

    crsco_data["sent"] = np.where(
        crsco_data.sent_x.isna(), crsco_data.sent_y, crsco_data.sent_x)

    crsco_data["APELLIDOS_PACIENTE"] = crsco_data.PacienteApPaterno.astype(
        str) + " " + crsco_data.PacienteApMaterno.astype(str)

    crsco_data.drop(['sent_x', 'sent_y', "PacienteApPaterno",
                    "PacienteApMaterno"], axis=1, inplace=True)

    crsco_data.rename(columns={"Attempt": "phone_call_answer_attemp",
                      "sent": "message_sent", "Comentario": "comentario"}, inplace=True)

    crsco_qualitative = crsco["qualitative"]

    crsco_qualitative = pd.concat([crsco_qualitative, crsco["qualitative_gc"]])

    crsco_data = crsco_data.merge(crsco_qualitative[["record", "action_received", "time", "helped", "obs",
                                  "motivo_no_show", "liked_method", "atendida"]], how="left", left_index=True, right_index=True)

    crsco_data.rename(columns={"motivo_no_show": "no_show_reason",
                      "atendida": "qualitative_interview_answered"}, inplace=True)

    hrt_data.columns = [column.lower() for column in hrt_data.columns]

    hlcm_data.columns = [column.lower() for column in hlcm_data.columns]

    crsco_data.columns = [column.lower() for column in crsco_data.columns]

    # Consolidation

    columns = list(hrt_data.columns) + \
        list(hlcm_data.columns) + list(crsco_data.columns)

    column_mapper = {


        "age_por": "agendado_por",


        "agendadopor": "agendado_por",


        "apellido_paciente": "apellidos_paciente",


        "celular": "telefono_movil",


        "telefono": "telefono_fijo",


        "telefono recado": "telefono_otro",


        "establecimientoinscripcion": "establecimiento_inscripcion",


        "establecimietoorigen": "establecimiento_origen",


        "codigoestablecimietoorigen": "codigo_establecimiento_origen",


        "atencionestablecidoentiempodeprotocolo": "atencion_establecido_en_tiempo_de_protocolo",


        "codigoprestacion": "codigo_prestacion",


        "codprestacion": "codigo_prestacion",


        "destinoalta": "destino_alta",


        "estadocita": "estado_cita",


        "estadoepisodio": "estado_episodio",


        "etapages": "etapa_ges",


        "fecha reserva": "fecha_reserva",


        "fechacita": "fecha_cita",


        "fechadenac": "fecha_nacimiento",


        "fechanac": "fecha_nacimiento",


        "fundamentodiagnostico": "fundamento_diagnostico",


        "horacita": "hora_cita",


        "hora_confirma": "hora_confirmacion",


        "leges": "le_ges",


        "motivoc": "motivo_cancelacion",


        "motivocancelacion": "motivo_cancelacion",


        "motivona": "motivo_no_atendido",


        "motivonoatendido": "motivo_no_atendido",


        "motivos": "motivo_sobrecupo",


        "motivosobrecupo": "motivo_sobrecupo",


        "numerole": "numero_le",


        "origen": "origen_cita",


        "origencita": "origen_cita",


        "pacientenombres": "nombres_paciente",


        "nombre_paciente": "nombres_paciente",


        "nombre_profesional": "nombres_profesional",


        "problemasalud": "problema_salud",


        "profesionalnombres": "nombres_profesional",


        "referenciasegunprotocolo": "referencia_segun_protocolo",


        "rut": "rut_paciente",


        "rut_profe": "rut_profesional",


        "rutpaciente": "rut_paciente",


        "rutprofesional": "rut_profesional",


        "tipoatencion": "tipo_atencion",


        "tipoprofesional": "tipo_profesional",


        "usu_confirma": "usuario_confirmacion",


        "usuario_confirma": "usuario_confirmacion",


    }

    data_consolidated = pd.concat([hrt_data.rename(columns=column_mapper), hlcm_data.rename(
        columns=column_mapper), crsco_data.rename(columns=column_mapper)], axis=0)

    boolean_mapper = {


        "Sí": True,


        "Si": True,


        "No": False,


        "SI": True,


        "NO": False,


        1: True,


        1.0: True,


        "1": True,


        "1.0": True,


        "ENVIADO": True,


        "ENVIADOS": True,


        "Listo": True,


        "LISTO": True,


        "listo": True,


        0: False,


        0.0: False,


        "0": False,


        "0.0": False,


        "NO ENVIADO": False,


        "Problemas con Plataforma": False,


        "RED FIJA": False,


        "SIN TELEFONO": False,


        "NO ENVIADO, PACIENTE CANCELA": False,


        "NO ENVIADO, PCTE CANCELA CITA": False,


        "NO ENVIADO, RED FIJA": False,


        "SOLO RED FIJA": False,


        "FALSO": False,


        "VERDADERO": True


    }

    sex_mapper = {


        "Mujer": "femenino",


        "F": "femenino",


        "Hombre": "masculino",


        "M": "masculino"


    }

    data_consolidated["action_received"] = data_consolidated.action_received.map(
        boolean_mapper)

    data_consolidated["fecha_nacimiento"] = pd.to_datetime(
        data_consolidated.fecha_nacimiento)

    data_consolidated["message_sent"] = data_consolidated.message_sent.map(
        boolean_mapper)

    data_consolidated["nsp"] = data_consolidated.nsp.map(boolean_mapper)

    data_consolidated["qualitative_interview_answered"] = np.where(
        data_consolidated.qualitative_interview_answered > 0, True, False)

    data_consolidated["record"] = data_consolidated.record.map(boolean_mapper)

    data_consolidated["sexo"] = data_consolidated.sexo.map(sex_mapper)

    data_consolidated.dropna(how="all", axis=1, inplace=True)

    establecimiento_abbreviator = {


        "TALCA, HOSPITAL REGIONAL DE": "hrt",


        "Centro de Referencia de Salud Cordillera Oriente": "crsco",


        "Hospital de Niños Dr. Luis Calvo Mackenna (Santiago, Providencia)": "hlcm"


    }

    data_consolidated["uid"] = data_consolidated.establecimiento.map(establecimiento_abbreviator) + "_" + data_consolidated.rut_paciente.apply(lambda x: hashlib.md5(str(x).encode()).hexdigest()[:5]) + "_" + data_consolidated.fecha_cita.dt.year.astype(
        str) + data_consolidated.fecha_cita.dt.month.astype(str) + data_consolidated.fecha_cita.dt.day.astype(str) + data_consolidated.hora_cita.apply(lambda x: x.hour).astype(str) + data_consolidated.hora_cita.apply(lambda x: x.minute).astype(str)

    data_consolidated = data_consolidated.dropna(
        subset=["uid"]).drop_duplicates(subset=["uid"])

    data_consolidated.set_index(
        "uid", verify_integrity=True, inplace=True, drop=True)
    return data_consolidated


if __name__ == '__main__':

    import argparse
    
    import pathlib

    parser = argparse.ArgumentParser()

    parser.add_argument('--hrt_db_url', help='HRT spreadsheet URL.')

    parser.add_argument('--hlcm_db_url', help='HLCM spreadsheet URL.')

    parser.add_argument('--crsco_db_url', help='CRSCO spreadsheet URL.')

    parser.add_argument('--crsco_state_url', help='CRSCO state spreadsheet URL.')
    
    parser.add_argument('--output_file', help='Output file destination.', type=pathlib.Path, default="data/processed/data_consolidated.csv")

    args = parser.parse_args()

    data_consolidated = get_data(
        hrt_db_url=args.hrt_db_url,

        hlcm_db_url=args.hlcm_db_url,

        crsco_db_url=args.crsco_db_url,

        crsco_state_url=args.crsco_state_url
    )
    
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    data_consolidated.to_csv(
        args.output_file, index=True)
