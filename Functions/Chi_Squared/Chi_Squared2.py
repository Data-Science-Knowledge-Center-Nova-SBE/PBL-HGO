def word_dummy(df, column, list, threshold = 90):

    from fuzzywuzzy import fuzz
    import re

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
        for i in df[column]:

            matches = re.findall(d, i)
            matches_count = len(matches)
            words_ref = i.split()
            number_words = len(words_ref)
            
            # # Find the similarity ratio between the word and each word in the text
            for text_word in i.split():
                
                ratio = fuzz.ratio(word, text_word)

                if ratio >= threshold:
                    matches_count += 1
            
            if number_words == 0:
                metric = 0

            else:
                metric = matches_count/number_words
            list_len.append(metric)
        
        df[column_name] = list_len  # df[column].str.contains(d)
        df[column_name] = df[column_name].astype(float)
    
    return df