def structured_data_dummies(alertP1):
    area = pd.get_dummies(alertP1['area'],drop_first=True)
    PROVENIENCIA = pd.get_dummies(alertP1['PROVENIENCIA'],drop_first=True)
    speciality_type = pd.get_dummies(alertP1['speciality_type'],drop_first=True)
    alertP1.drop(['area','PROVENIENCIA','speciality_type'],axis=1,inplace=True)
    alertP1 = pd.concat([data,area,PROVENIENCIA,speciality_type],axis=1)
    
    return alertP1