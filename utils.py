"""
Holds useful functions until there's a more specific place to put them
"""

import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

import re
import string

lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

#################################################
#
#     Pre-processing
#
#################################################

def preprocess(doc):
    doc = doc.lower().replace(u'•','')
    doc = remove_punctuation(doc)
    doc = " ".join([w for w in word_tokenize(doc) if not w in stop_words])
    return doc

def remove_punctuation(text):
    return re.sub('[' + string.punctuation + u'’“”' +  ']', "", text)

def df_to_stringlist(df, apply_preprocess=True):
    docs = df['content'].tolist()
    return [preprocess(doc) for doc in docs] if apply_preprocess else docs

def simple_word_counts(prepared_doc, n=20):
    tokens = word_tokenize(prepared_doc)
    counts = Counter(tokens)
    return counts.most_common(n)

#################################################
#
#     Analysis
#
#################################################

def tokenize(text):
    return [lemmatizer.lemmatize(w) for w in word_tokenize(text)]

def vec_output_to_df(vectorizer, output):
    # index is the index of the story
    return pd.DataFrame(data=output.toarray(), columns=vectorizer.get_feature_names())

def get_top_by_idx(df, doc_idx, n=10):
    return df.T[doc_idx].sort_values(ascending=False)[:n]


##### LDA - DUMP THIS HERE

from sklearn.decomposition import LatentDirichletAllocation


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


def run_lda(docs, n_components=2, n_top_words=10):
    tf_vectorizer = CountVectorizer(tokenizer=tokenize, max_df=0.95, min_df=2)
    tf = tf_vectorizer.fit_transform(docs)

    print("Fitting LDA models with tf features, "
          "n_components=%d..."
          % (n_components))
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=10,
                                    learning_method='online',
                                    learning_offset=50.)

    lda.fit(tf)

    print_top_words(lda, tf_vectorizer.get_feature_names(), n_top_words)


#################################################
#
#     Plot
#
#################################################

def make_series_bar(ax, series, title, ylabel=None):
    x = range(len(series))
    ax.set_title(title)
    ax.bar(x, series)
    ax.set_ylabel(ylabel)
    plt.sca(ax)
    plt.xticks(x, series.index, rotation=45)
    return ax