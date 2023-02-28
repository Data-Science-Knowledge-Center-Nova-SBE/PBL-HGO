def class_outcome(alertP1):
    alertP1['result']=['1' if x in [0,14,15,53,8,12,13] else '0' if x in [1,6,10,7] else '' for x in alertP1['COD_MOTIVO_RECUSA'] ]
    alertP1 = alertP1[alertP1['result']!='']
    return alertP1

def class_area(alertP1):
    area_list=[3150502,3151672,3150671,3150572,3150305,3150571,3151571,3151574,3150371,3151573,3150672,3152401,3150605,3152471,3151575,3151671,3151601,3150573,3151576,3150506,3150504,3152400,3150501,3150603,3150604,3151707,999999,3152403,3151603,3151500,3150503,3150330,3150600]
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

def unit(alertP1):
    USF_list=[2090771,3111172,3111174,3112271,3113271,3113274,3113672,3113871,3114272,3114471,3150371,3150571,3150572,3150573,3150872,3151571,3151572,3151573,3151574,3151575,3151576,3151671,3151672,3151771,3151772,3151871,3151872,3152403,3152471,3152671,4070571,4070671,4121571]
    UCSP_list=[3150305,3150506,3151701,3151707,3151802,3152401]
    alertP1['unit']=['USF' if x in USF_list else 'UCSP' if x in UCSP_list else 'CS' for x in alertP1['COD_UNID_SAUDE_PROV']]
    alertP1['unit'][alertP1['TIPO_UNID']!='CS/USF']='HOSP'
    return alertP1

