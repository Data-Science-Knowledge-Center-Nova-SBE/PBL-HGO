import pandas as pd
def structured_data_dummies(alertP1):
    data=alertP1.copy()
    area = pd.get_dummies(data['area'],drop_first=True)
    Provenance = pd.get_dummies(data['PROVENIENCIA'],drop_first=True)
    speciality = pd.get_dummies(data['DES_ESPECIALIDADE'],drop_first=True)
    step= pd.get_dummies(data['step'],drop_first=True)
    unit= pd.get_dummies(data['unit'],drop_first=True)
    data.drop(['area','PROVENIENCIA','speciality_type','step','unit','DES_ESPECIALIDADE'],axis=1,inplace=True)
    data = pd.concat([data,area,Provenance,speciality,step,unit],axis=1)
    return data
