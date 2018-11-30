#!/usr/bin/env python
# coding: utf-8

# ![BTS](img/Logo-BTS.jpg)
# 
# # Session 5: Text Mining (II)
# 
# ### Juan Luis Cano Rodríguez <juan.cano@bts.tech> - Data Science Foundations (2018-10-19)
# 
# Open this notebook in Google Colaboratory: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Juanlu001/bts-mbds-data-science-foundations/blob/master/sessions/05-Text-Mining-II.ipynb)

# In[1]:


# Source: http://billchambers.me/tutorials/2014/12/21/tf-idf-explained-in-python.html

t0 = "China has a strong economy that is growing at a rapid pace. However politically it differs greatly from the US Economy."
t1 = "At last, China seems serious about confronting an endemic problem: domestic violence and corruption."
t2 = "Japan's prime minister, Shinzo Abe, is working towards healing the economic turmoil in his own country for his view on the future of his people."
t3 = "Vladimir Putin is working hard to fix the economy in Russia as the Ruble has tumbled."
t4 = "What's the future of Abenomics? We asked Shinzo Abe for his views"
t5 = "Obama has eased sanctions on Cuba while accelerating those against the Russian Economy, even as the Ruble's value falls almost daily."
t6 = "Vladimir Putin was found to be riding a horse, again, without a shirt on while hunting deer. Vladimir Putin always seems so serious about things - even riding horses."


# ## Exercise 1: Jaccard similarity
# 
# 1. Write a function `lemmatize` that receives a spaCy `Doc` and returns a list of lemmas as strings, removing stopwords, punctuation signs and whitespace
# 2. Write a function that receives two spaCy `Doc`s and returns a floating point number representing the Jaccard similarity (see formula below) (hint: use [`set`s](https://docs.python.org/3/library/stdtypes.html#set))
# 3. Compute the Jaccard similarity between `t0` and `t1`
# 4. Create a pandas `DataFrame` that holds the Jaccard similarity of all the text combinations from `t0` to `t6` (hint: use [`enumerate`](http://book.pythontips.com/en/latest/enumerate.html#enumerate))
# 
# $$ J(A,B) = {{|A \cap B|}\over{|A \cup B|}} $$

# In[1]:


import spacy
from spacy.lang.en.stop_words import STOP_WORDS


# In[2]:


nlp = spacy.load("en_core_web_sm")


# In[4]:


def lemmatize(doc):
    return [token.lemma_ for token in doc if not
           (token.is_punct or token.is_space or token.lower_ in STOP_WORDS)]


# In[5]:


lemmatize(nlp(t0))


# In[6]:


def jaccard(doc1, doc2):
    s1 = set(lemmatize(doc1))
    s2 = set(lemmatize(doc2))
    return len(s1.intersection(s2)) / len(s1.union(s2))


# In[7]:


jaccard(nlp(t0), nlp(t1))


# In[8]:


import numpy as np
import pandas as pd


# In[9]:


data = np.zeros((7, 7))
docs = [nlp(text) for text in (t0, t1, t2, t3, t4, t5, t6)]
for ii, doc_a in enumerate(docs):
    for jj, doc_b in enumerate(docs):
        data[ii, jj] = jaccard(doc_a, doc_b)

pd.DataFrame(data)


# ## Exercise 2: TF-IDF
# 
# 1. Write a function `tf` that receives a string and a spaCy `Doc` and returns the number of times the word appears in the `lemmatize`d `Doc`
# 2. Write a function `idf` that receives a string and a list of spaCy `Doc`s and returns the number of docs that contain the word
# 3. Write a function `tf_idf` that receives a string, a spaCy `Doc` and a list of spaCy `Doc`s and returns the product of `tf(t, d) · idf(t, D)`.
# 4. Write a function `all_lemmas` that receives a list of `Doc`s and returns a `set` of all available `lemma`s
# 5. Write a function `tf_idf_doc` that receives a `Doc` and a list of `Doc`s and returns a dictionary of `{lemma: TF-IDF value}`, corresponding to each the lemmas of all the available documents
# 6. Write a function `tf_idf_scores` that receives a list of `Doc`s and returns a `DataFrame` displaying the lemmas in the columns and the documents in the rows.
# 7. Visualize the TF-IDF, like this:
# 
# ![TF-IDF](img/tf-idf.png)

# In[3]:


import spacy
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load('en_core_web_sm')


# In[3]:


from collections import Counter


# In[89]:


def lemmatize(doc):
    return [
        token.lemma_
        for token in doc
        if not token.is_space and not token.is_punct and not token.lower_ in STOP_WORDS
        and not token.tag_ == "POS"
    ]


# In[20]:


def tf(word, doc):
    lemmas = lemmatize(doc)
    #return Counter(lemmas)[word]
    return lemmas.count(word)


# In[59]:


def tf(word, doc):
    lemmas = lemmatize(doc)
    #return Counter(lemmas)[word]
    print(lemmas.count(word))


# In[61]:


def tf(word, doc):
    lemmas = lemmatize(doc)
    return Counter(lemmas)[word]


# In[51]:


count = 2
if count:
    print(1 / count)
else:
    print(0)


# In[72]:


def foo(a):
    return a > 0


# In[73]:


foo(1)


# In[75]:


foo(-1)


# In[69]:


def foo(a):
    condition = a > 0
    if condition:
        return True
    else:
        return False


# In[70]:


foo(1)


# In[71]:


foo(-1)


# In[64]:


if "":
    print("Empty string is true")


# In[67]:


if [0]:
    print("Empty list is true")


# In[63]:


if 0:
    print("Zero is true")


# In[42]:


def idf(word, docs):
    count = 0
    for doc in docs:
        if word in lemmatize(doc):
            count += 1

    # We don't need to account for the 0 case since all the words will be in at least 1 document
    return 1 / count  # if count else 0


# In[40]:


docs = [nlp(text) for text in (t0, t1, t2, t3, t4, t5, t6)]


# In[41]:


idf("china", docs)


# In[43]:


def tf_idf(word, doc, docs):
    return tf(word, doc) * idf(word, docs)


# In[44]:


tf_idf("china", nlp(t1), docs)


# In[90]:


def all_lemmas(docs):
    lemmas = set()
    for doc in docs:
        #lemmas = lemmas.union(set(lemmatize(doc)))
        #lemmas = set(lemmatize(doc))
        lemmas.update(set(lemmatize(doc)))

    return lemmas


# In[91]:


print(all_lemmas(docs))


# In[98]:


def tf_idf_doc(doc, docs):
    lemmas = all_lemmas(docs)
    values = {}
#     for lemma in lemmas:
#         values[lemma] = tf_idf(lemma, doc, docs)

#     return values
    return {lemma: tf_idf(lemma, doc, docs) for lemma in lemmas}


# In[100]:


print(tf_idf_doc(nlp(t1), docs))


# In[102]:


import pandas as pd


# In[104]:


ll = ['a', 'b', 'c']
ll.append('d')
ll


# In[109]:


ll = ['a', 'b', 'c']
ll + ['d', 'e']


# In[106]:


ll = ['a', 'b', 'c']
ll.extend(['d', 'e'])
ll


# In[101]:


def tf_idf_scores(docs):
    rows = []
    for doc in docs:
        rows.append(tf_idf_doc(doc, docs))
    
    return pd.DataFrame(rows)


# In[110]:


res = tf_idf_scores(docs)
res


# In[112]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[114]:


import seaborn as sns


# In[118]:


fig, ax = plt.subplots(figsize=(15, 3))
sns.heatmap(res, ax=ax)


# In[119]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[113]:


plt.pcolor(res)


# In[10]:


from collections import Counter


# In[11]:


def tf(word, doc):
    counts = Counter(lemmatize(doc))
    return counts[word]


# In[12]:


tf('economy', nlp(t0))


# In[13]:


def idf(word, docs):
    count = 0
    for doc in docs:
        if word in lemmatize(doc):
            count += 1
    return 1 / count


# In[14]:


idf('economy', docs)


# In[15]:


def tf_idf(word, doc, docs):
    assert doc in docs
    return tf(word, doc) * idf(word, docs)


# In[16]:


tf_idf('economy', docs[0], docs)


# In[17]:


def all_lemmas(docs):
    lemmas = set()
    for doc in docs:
        lemmas.update(lemmatize(doc))
    return lemmas


# In[18]:


print(all_lemmas(docs))


# In[19]:


def tf_idf_doc(doc, docs):
    lemmas = all_lemmas(docs)
    res = {}
    for lemma in lemmas:
        res[lemma] = tf_idf(lemma, doc, docs)
    return res


# In[20]:


print(tf_idf_doc(docs[0], docs))


# In[21]:


def tf_idf_scores(docs):
    lemmas = all_lemmas(docs)
    rows = []
    for doc in docs:
        rows.append(tf_idf_doc(doc, docs))

    return pd.DataFrame(rows)


# In[22]:


tb = tf_idf_scores(docs)
tb


# In[23]:


tb / tb.max().max()


# In[24]:


tb.columns


# In[25]:


tb


# In[32]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(15, 3))
sns.heatmap(tb / tb.max().max(), cmap="RdYlGn_r", annot=False);

