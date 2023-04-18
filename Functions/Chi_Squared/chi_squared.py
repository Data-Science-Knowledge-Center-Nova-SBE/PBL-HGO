import re
import pandas as pd

def word_dummy(df, column):
    
    list =  ['tac',
             'ce',
             'ter',
             'mês',
             'mg',
             'algum',
             'sintomatologia',
             'outro',
             'analise',
             'tremor',
             'ligeiro',
             'dificuldade',
             'avaliacao',
             'tac\\ ce',
             'alteracoe',
             'vez',
             'repouso']

    chi2_columns_2 = []

    for word in list:
        word2 = re.escape(word)
        word_regex = '([\W\s]{1}' + word2 + '[\W\s]{1})'
        chi2_columns_2.append(word_regex)
        
    for d in chi2_columns_2:
        column_name = d.replace('([\W\s]{1}', '')
        column_name = column_name.replace('[\W\s]{1})', '')
        df[column_name] = df[column].str.contains(d)
        df[column_name] = df[column_name].astype(int)

    return df