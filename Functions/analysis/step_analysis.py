def step_acceptance_rate(alertP1):
    accepted=alertP1[alertP1['result']=='1']['ID_DOENTE'].groupby(alertP1['step']).count().reset_index().rename(columns={"ID_DOENTE":"number_of_acceptance"})
    total_referrals = alertP1['ID_DOENTE'].groupby(alertP1['step']).count().reset_index().rename(columns={"ID_DOENTE":"number_of_referrals"})
    #divide the number of referrals for each specialty and priority level by the total number of referrals for that specialty, then multiply by 100 to get the percentage
    accepted['percentage'] = round(accepted['number_of_acceptance'] / total_referrals['number_of_referrals'] * 100)
    return(accepted)