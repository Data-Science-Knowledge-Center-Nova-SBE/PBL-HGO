def remove_stopwords(text_list):
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    # Get the Portuguese stop words
    stop_words = set(stopwords.words('portuguese'))
    stop_words.update(['.', ',','(',')',':','-','?','+','/',';','drª','``','','desde','doente','consulta','alterações','se',"''",'cerca','refere','hgo','utente','vossa','s','...','ainda','c','filha','costa','dr.','pereira','ja','--','p','dr','h','n','>','q','//','..','b','++','%'])

    # Create a new list to store the filtered text
    filtered_text = []

    # Loop through the text_list and remove the stop words
    for text in text_list:
        words = word_tokenize(text)
        words = [word for word in words if word.lower() not in stop_words]
        filtered_text.append(" ".join(words))

    return filtered_text