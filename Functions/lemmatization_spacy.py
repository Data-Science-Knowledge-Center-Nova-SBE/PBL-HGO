def spacy_lemmatizer(text_list):
    # Spacy is required
    # $pip install -U spacy
    # $python -m spacy download pt_core_news_md
    # Additional information: https://spacy.io/usage
    
    import spacy
    import pt_core_news_md
    nlp = pt_core_news_md.load()

    doclist = list(nlp.pipe(text_list))

    docs=[]
    for i, doc in enumerate(doclist):
        docs.append(' '.join([listitem.lemma_ for listitem in doc]))
        
    return docs