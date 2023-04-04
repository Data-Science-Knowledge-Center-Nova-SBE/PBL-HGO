#create a column with area of the unit
def class_area(alertP1):
    #create a list with code of units inside area
    area_list=[3150502,3151672,3150671,3150572,3150305,3150571,3151571,3151574,3150371,3151573,3150672,3152401,3150605,3152471,3151575,3151671,3151601,3150573,3151576,3150506,3150504,3152400,3150501,3150603,3150604,3151707,999999,3152403,3151603,3151500,3150503,3150330,3150600]
    alertP1['area']=['inside area' if x in area_list else 'outside area' for x in alertP1['COD_UNID_SAUDE_PROV'] ]
    return alertP1

#speciality 
def speciality(alertP1):
    alertP1['speciality_type'] = ['General Neurology' if x == 'NEUROLOGIA' else 'Other specialities'  for x in alertP1['DES_ESPECIALIDADE']]
    return(alertP1)
#compute length of text
def text_length(alertP1):
    alertP1['text_length']=alertP1['Texto'].str.len()
    return alertP1

#step of referral(first or second or3+)
def referral_steps(alertP1):
    alertP1['step']=alertP1.sort_values(by=['DATA_RECEPCAO']).groupby('ID_DOENTE').cumcount()+1
    alertP1['step'][alertP1['step']>=3]='3+'
    return alertP1


#create a column with units
def unit(alertP1):
    #create  lists with units 
    USF_list=[2090771,3111172,3111174,3112271,3113271,3113274,3113672,3113871,3114272,3114471,3150371,3150571,3150572,3150573,3150872,3151571,3151572,3151573,3151574,3151575,3151576,3151671,3151672,3151771,3151772,3151871,3151872,3152403,3152471,3152671,4070571,4070671,4121571]
    UCSP_list=[3150305,3150506,3151701,3151707,3151802,3152401]
    alertP1['unit']=['USF' if x in USF_list else 'UCSP' if x in UCSP_list else 'CS' for x in alertP1['COD_UNID_SAUDE_PROV']]
    alertP1['unit'][alertP1['TIPO_UNID']!='CS/USF']='HOSP'
    return alertP1


#the code sorts the referrals first for ID and after data _recepçao.
# the first referral is always 0 even it is accepted or rejected. 
#If there is a referral acceptad before than all other referrals of that patient is 1
def bef_accepted(alertP1):

    df_sorted = alertP1.sort_values(['ID_DOENTE', 'DATA_RECEPCAO'])
    df_sorted["result"]=df_sorted["result"].astype("int")
    df_sorted.loc[df_sorted['ID_DOENTE'].ne(df_sorted['ID_DOENTE'].shift()), 'before_accepted'] = 0
    df_sorted['before_accepted'] = df_sorted.groupby('ID_DOENTE')['result'].cumsum().clip(upper=1)
    df_sorted.loc[df_sorted.groupby('ID_DOENTE').head(1).index, 'before_showed_up'] = 0
    return alertP1











