def word_dummy(df, column):
<<<<<<< HEAD

    from fuzzywuzzy import fuzz
    import re

    list = ['nao',
                'ha',
                'medicar',
                'ce',
                'tac',
                'familiar',
                'hta',
                'alteracoes',
                'fazer',
                'agravamento',
                'avaliacao',
                'quadro',
                'ter',
                'ano',
                'alteracoe',
                'neurologia',
                'mês',
                'dra',
                'ap',
                'realizar',
                'memoria']

    # selected_words = ['medicar','ce','tac','familiar','hta','alteracoes','fazer','agravamento','avaliacao','quadro','ano','alteracoe','neurologia','dra','realizar','memoria','Medicina','pedir','geral','exame','antecedente','apresentar','episodio','queixa','terapeutico','cognitivo','sintomatologia','frequente','esquerdo','revelar','tc','cefaleia','actual','historia','problema','cerebral','avc','sindrome','progressivo','ligeiro','doenca','dislipidemia','demencial','vascular','tremor','clinico','demencia','dm','frontal','direito','referir','observacao','neurologico','iniciar','temporal','pos','cronico','agravar','orientacao','altura','evolucao','lesao','isquemico','medico','provavel','bilateral','desorientacao','dta','moderar','comportamento','atrofia']
=======
    
    from fuzzywuzzy import fuzz
    import re
    df['chi_squared']=df[column]
    #list = ['nao','ha','medicar','ce','tac','familiar','hta','alteracoes','fazer','agravamento','avaliacao','quadro','ter','ano','alteracoe','neurologia','mês','dra','ap','realizar','memoria']

    list = ['medicar','ce','tac','familiar','hta','alteracoes','fazer','agravamento','avaliacao','quadro','ano','alteracoe','neurologia','dra','realizar','memoria','Medicina','pedir','geral','exame','antecedente','apresentar','episodio','queixa','terapeutico','cognitivo','sintomatologia','frequente','esquerdo','revelar','tc','cefaleia','actual','historia','problema','cerebral','avc','sindrome','progressivo','ligeiro','doenca','dislipidemia','demencial','vascular','tremor','clinico','demencia','dm','frontal','direito','referir','observacao','neurologico','iniciar','temporal','pos','cronico','agravar','orientacao','altura','evolucao','lesao','isquemico','medico','provavel','bilateral','desorientacao','dta','moderar','comportamento','atrofia']
>>>>>>> origin/marouan

    unique_words_regex = []

    for word in list:
        word2 = re.escape(word)
        word_regex = '([\W\s]{1}' + word2 + '[\W\s]{1})'
        unique_words_regex.append(word_regex)
    
    
    # Create a new column for each word in the list. d is the unique word with the regex
    for d in unique_words_regex:
        
        list_len = []
        # Create a new column name, remove the regex
        column_name = d.replace('([\W\s]{1}', '')
        column_name = column_name.replace('[\W\s]{1})', '')

        # Loop through the column and check if the word is in the text
<<<<<<< HEAD
        for i in df[column]:
=======
        for i in df['chi_squared']:
>>>>>>> origin/marouan

            matches = re.findall(d, i)
            matches_count = len(matches)
            words_ref = i.split()
            number_words = len(words_ref)
            
            
            if number_words == 0:
                metric = 0

            else:
                metric = matches_count/number_words
            list_len.append(metric)
        
<<<<<<< HEAD
        df[column_name] = list_len  # df[column].str.contains(d)
        df[column_name] = df[column_name].astype(float)
    
    return df
=======
        df[column_name] = list_len  
        df[column_name] = df[column_name].astype(float)
    
    return df
def chi_squared(alertP1):
    word_dummy(alertP1,"clean_text")
    return(alertP1)
>>>>>>> origin/marouan
