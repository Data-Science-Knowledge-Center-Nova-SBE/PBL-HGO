import pandas as pd
def structured_data_dummies(alertP1):
    data=alertP1.copy()
    area = pd.get_dummies(data['area'],drop_first=True)
    Provenance = pd.get_dummies(data['PROVENIENCIA'],drop_first=True)
    speciality = pd.get_dummies(data['speciality_type'],drop_first=True)
<<<<<<< HEAD
    step= pd.get_dummies(data['step'],drop_first=True)
    unit= pd.get_dummies(data['unit'],drop_first=True)
    data.drop(['area','PROVENIENCIA','step','unit','speciality_type'],axis=1,inplace=True)
    data = pd.concat([data,area,Provenance,speciality,step,unit],axis=1)
    data.columns = data.columns.astype(str)
=======
    before_accepted = pd.get_dummies(data['before_accepted'],drop_first=True)
    step= pd.get_dummies(data['step'],drop_first=True)
    unit= pd.get_dummies(data['unit'],drop_first=True)
    data.drop(['area','PROVENIENCIA','step','unit','speciality_type','before_accepted'],axis=1,inplace=True)
    data = pd.concat([data,area,Provenance,speciality,step,unit,before_accepted],axis=1)
>>>>>>> gabrabib
    return data

