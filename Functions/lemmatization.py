def lem_pt(text):
    import spacy
    import pt_core_news_md
    nlp = pt_core_news_md.load()

    #Lemmatization Output
    document = nlp(text)
    lemmatized_spacy_output = " ".join([token.lemma_ for token in document])
    print(lemmatized_spacy_output)



#EXAMPLE
texto = "Guilherme Silva, 21 anos, estuda direito, sofre de enxaquecas graves e insónias durante a noite. Os épisódios acontecem há 3 meses, duas a três vezes por semana. Toma benuron mas não faz efeito. A situação manteve-se estável ao início mas agora existe um agravamento"
lem_pt(texto)

# THIS IS WORKING ONLY FOR INDIVIDUAL STRINGS