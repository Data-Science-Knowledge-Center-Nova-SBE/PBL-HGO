def date_format_alertP1(alertP1):
    import pandas as pd

    alertP1["DATA_ENVIO"] = pd.to_datetime(alertP1["DATA_ENVIO"],dayfirst=True, yearfirst=False )
    alertP1["DATA_RECEPCAO"] = pd.to_datetime(alertP1["DATA_RECEPCAO"],dayfirst=True, yearfirst=False)
    alertP1["DATA_RETORNO"] = pd.to_datetime(alertP1[alertP1["DATA_RETORNO"]!='26/06/0214']["DATA_RETORNO"],dayfirst=True, yearfirst=False)
    alertP1["DATA_REALIZACAO"] = pd.to_datetime(alertP1["DATA_REALIZACAO"],dayfirst=True, yearfirst=False)
    alertP1["DATA_MARCACAO"] = pd.to_datetime(alertP1["DATA_MARCACAO"],dayfirst=True, yearfirst=False)
    return alertP1

def replace_blank(alertP1):
    import pandas as pd
    
    alertP1['PROVENIENCIA'][alertP1['PROVENIENCIA']=='']='unknown'
    alertP1['CTH_PRIOR'][alertP1['CTH_PRIOR']=='']='unknown'
    return alertP1

