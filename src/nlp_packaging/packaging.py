import pandas as pd

from spacy.lang.en.stop_words import STOP_WORDS


def lemmatize(doc):
    return [
        token.lemma_
        for token in doc
        if not token.is_space and not token.is_punct and not token.lower_ in STOP_WORDS
        and not token.tag_ == "POS"
    ]


def tf(word, doc):
    lemmas = lemmatize(doc)
    #return Counter(lemmas)[word]
    return lemmas.count(word)


def idf(word, docs):
    count = 0
    for doc in docs:
        if word in lemmatize(doc):
            count += 1

    # We don't need to account for the 0 case since all the words will be in at least 1 document
    return 1 / count  # if count else 0


def tf_idf(word, doc, docs):
    return tf(word, doc) * idf(word, docs)


def all_lemmas(docs):
    lemmas = set()
    for doc in docs:
        #lemmas = lemmas.union(set(lemmatize(doc)))
        #lemmas = set(lemmatize(doc))
        lemmas.update(set(lemmatize(doc)))

    return lemmas


def tf_idf_doc(doc, docs):
    lemmas = all_lemmas(docs)
    values = {}
#     for lemma in lemmas:
#         values[lemma] = tf_idf(lemma, doc, docs)

#     return values
    return {lemma: tf_idf(lemma, doc, docs) for lemma in lemmas}


def tf_idf_scores(docs):
    rows = []
    for doc in docs:
        rows.append(tf_idf_doc(doc, docs))

    return pd.DataFrame(rows)
