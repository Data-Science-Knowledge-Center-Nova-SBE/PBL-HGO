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
    USF_list=[2090771, 3111172, 3111174, 3112271, 3113271, 3113274, 3113672, 3113871, 3114272, 3114471, 3150109, 3150371, 3150371, 3150501, 3150502,3150571, 3150572, 3150573, 3150600, 3150603, 3150604, 3150671, 3150672, 3150772, 3150773, 3150872, 3151104, 3151400,3151407, 3151409, 3151500, 3151571, 3151572, 3151573, 3151574, 3151575, 3151576, 3151601, 3151671, 3151672, 3151771, 3151772, 3151871, 3151872, 3152001, 3152002, 3152403, 3152471, 3152671, 4070671, 4120400, 4121300, 4121571]
    UCSP_list=[3111400, 3111800, 3140500, 3150301, 3150305, 3150504, 3150506, 3150605, 3150801, 3151000, 3151101, 3151201, 3151306, 3151402, 3151404, 3151406, 3151600, 3151701, 3151707, 3151802, 3152100, 3152400, 3152401, 4020200, 4020604, 4021100, 4021104, 4021108, 4021110, 4021112, 4021113, 4070400, 4071200, 4120100, 4120602, 4121100, 4121101, 4121200, 4121400, 4121500]
    alertP1['unit']=['USF' if x in USF_list else 'UCSP' if x in UCSP_list else 'HGO' if x==0 else 'HOSP' for x in alertP1['COD_UNID_SAUDE_PROV']]
    return alertP1













