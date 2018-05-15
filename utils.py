"""
Holds useful functions until there's a more specific place to put them
"""

import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
# from nltk.stem.porter import
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

import re
import string

lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer("english")
stop_words = stopwords.words('english')


#################################################
#
#     Convenience methods
#
#################################################

def get_idx_from_id(df, sid):
    return df.index.get_loc(df[df.id == sid].index[0])


def filter_n_highest(arr, n_highest):
    return arr.argsort()[-n_highest:][::-1]


#################################################
#
#     Pre-processing
#
#################################################

def preprocess(doc):
    doc = doc.lower().replace(u'•', '')
    doc = remove_punctuation(doc)
    doc = " ".join([w for w in word_tokenize(doc) if not w in stop_words])
    return doc


def remove_punctuation(text):
    text.replace('-' ,' ') # not everything can reasonably be removed. dashes ought to be replaced by space
    return re.sub('[' + string.punctuation + u'’“”' + ']', "", text)


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

def tokenize_stem(text):
    return [stemmer.stem(w) for w in word_tokenize(text)]

def vec_output_to_df(vectorizer, output):
    # index is the index of the story
    return pd.DataFrame(data=output.toarray(), columns=vectorizer.get_feature_names())


def get_top_by_idx(df, doc_idx, n=10):
    """ For df of term freq """
    return df.T[doc_idx].sort_values(ascending=False)[:n]


def get_events_daterange(df, dmy_date, day_range=7):
    day, month, year = dmy_date
    return df[(df.year == year) & (df.month == month) & (df.day >= day) & (df.day < day + day_range)]


def get_cosine_similarity(docs, max_df=0.95, min_df=2):
    """ Can take either list of precessed docs or df of stories and it will do the processing step"""

    if type(docs) is pd.DataFrame:
        print('Converting dataframe to stringlist')
        docs = df_to_stringlist(docs)

    print('Calculating cosine similarity using tfidf: max_df={}, min_df={}'.format(max_df, min_df))
    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, max_df=max_df, min_df=min_df)  # max_features=n_features,
    # note: min_df and max_df params: If float, the parameter represents a proportion of documents; if integer, absolute counts.
    tfidf = tfidf_vectorizer.fit_transform(docs)

    #     tf_vectorizer = CountVectorizer(tokenizer=tokenize, max_df=0.95, min_df=2)
    #     tf = tf_vectorizer.fit_transform(docs)
    #     labelled_counts = vec_output_to_df(tf_vectorizer, tf)
    print('TFIDF vectorization complete. Calculating cosine similarity...')
    return cosine_similarity(tfidf)


def kmeans_split(df, order_by_sid=None):
    """ Note: we don't know which class will be which!"""
    docs = df_to_stringlist(df)

    tf_vectorizer = CountVectorizer(tokenizer=tokenize)
    tf = tf_vectorizer.fit_transform(docs)

    kmeans = KMeans(n_clusters=2).fit(tf)
    c0, c1 = df[kmeans.labels_ == 0], df[kmeans.labels_ == 1]

    if order_by_sid is not None:
        classes = (c0, c1) if not c0[c0.id == order_by_sid].empty \
            else (c1, c0) if not c1[c1.id == order_by_sid].empty \
            else None
    else:
        classes = (c1,c1)

    if classes is None:
        print(' == No clear ordering. Invalid sid?? == ')

    return classes


def get_most_similar(df, cs_matrix, story_id, n_most_similar=10):
    primary_idx = get_idx_from_id(df, story_id)
    return df.iloc[filter_n_highest(cs_matrix[primary_idx], n_most_similar)]

def get_similar_by_daterange_sid(df, date, sid, n_most_similar=30, day_range=7):
    print('Getting stories within {} days of '.format(day_range) + '{1}/{0}/{2} (within the same month)'.format(*list(date)))
    df_range = get_events_daterange(df, date, day_range)
    print('Processing stories...')
    stories = df_to_stringlist(df_range)
    cs = get_cosine_similarity(stories)

    return get_most_similar(df_range, cs, story_id=sid, n_most_similar=n_most_similar)

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
