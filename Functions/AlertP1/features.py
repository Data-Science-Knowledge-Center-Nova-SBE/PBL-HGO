def class_outcome(alertP1):
    alertP1['result']=['Accepted' if x in [0,14,15,53,8,12,13] else 'Refused' if x in [1,6,10,7] else '' for x in alertP1['COD_MOTIVO_RECUSA'] ]
    alertP1 = alertP1[alertP1['result']!='']
    return alertP1

def class_area(alertP1):
    area_list=[3150502,3151672,3150671,3150572,3150305,3150571,3151571,3151574,3150371,3151573,3150672,3152401,3150605,3152471,3151575,3151671,3151601,3150573,3151576,3150506,3150504,3152400,3150501,3150603,3151707,9999999,3152403,3151400,3152100,3151401,4021100,4021104,3152002]
    alertP1['area']=['inside area' if x in area_list else 'outside area' for x in alertP1['COD_UNID_SAUDE_PROV'] ]
    return alertP1

def class_speciality(alertP1):
    alertP1['speciality_type'] = ['General Neurology' if x == 'NEUROLOGIA' else 'Other specialities'  for x in alertP1['DES_ESPECIALIDADE']]
    return alertP1

def text_lenght(alertP1):
    alertP1['text_length']=alertP1['Texto'].str.len()
    return alertP1

def referral_steps(alertP1):
    alertP1['step']=alertP1.sort_values(by=['DATA_RECEPCAO']).groupby('ID_DOENTE').cumcount()+1
    return alertP1

