import re
from collections import Counter
import numpy as np

def remove_punctuation_regex(text):
    text = text.replace("n't", " not") # deal with contractions
    text = re.sub("\'m|\'ll|\'s|\'d", " ", text)
    text = text.replace("'", " ")
    return re.sub(r'[^\w\s\n\t]', '', text)

def tokenize(text):
    text_vec = [e.strip().lower() for e in text.split(" ")]
    text_vec = [e for e in text_vec if e != ""]
    return text_vec

def create_bag_of_words(corpus, df_min, df_max_proportion, tfidf_threshold_q, tokenized=False):
    '''
    INPUT:
    corpus: if tokenized=False, a list of dictionaries: each inner dictionary maps each word to its count in the document;
    if tokenized=True, a list of numpy arrays
    df_min: minimum # of document frequency to keep a word in vocabulary
    df_max_proportion: maximum document frequency (in proportions) to keep a word in vocabulary (stop words removal)
    tfidf_threshold_q: the threshold tf-idf quantile to keep a word in a document
    OUTPUT:
    word_counts_by_document: the updated corpus after fitering by df and tf-idf
    vocab_new: list of all unique words in the corpus after filtering
    n: total number of words in the updated corpus
    '''
    if tokenized == True:
        word_counts_by_document = [dict(Counter(rv)) for j,rv in enumerate(corpus)]
    else:
        word_counts_by_document = [dict(Counter(tokenize(rv))) for j,rv in enumerate(corpus)]
    document_length = [len(rv) for rv in corpus]
    tf = {} # term frequency
    for k,v in enumerate(word_counts_by_document):
        tf[k] = {}
        for k2,v2 in v.items():
            tf[k][k2] = v2/document_length[k]
    df = {} # document frequency
    for k,v in tf.items():
        for k2 in v:
            try:
                df[k2] += 1
            except KeyError:
                df[k2] = 1
    vocab = set()
    for k,v in df.items():
        if v/len(corpus) <= df_max_proportion and v >= df_min: # remove extremely rare words and stop words
            vocab.add(k)
    idf = {k:np.log(len(corpus)/v) for k,v in df.items() if k in vocab} # inverse document frequency
    tfidf = {} # TF-IDF before filtering
    for k,v in tf.items():
        tfidf[k] = {}
        for k2,v2 in v.items():
            if k2 in vocab:
                tfidf[k][k2] = v2 * idf[k2]
    all_tfidf_vals = []
    for k,v in tfidf.items():
        for k2,v2 in v.items():
            all_tfidf_vals.append(v2)
    threshold_tfidf = np.quantile(all_tfidf_vals, tfidf_threshold_q)
    tfidf_new = {} # TF-IDF after filtering
    for k,v in tfidf.items():
        tfidf_new[k] = {}
        for k2,v2 in v.items():
            if v2 >= threshold_tfidf:
                tfidf_new[k][k2] = v2
    n = 0
    vocab_new = set()
    for k,v in tfidf_new.items():
        to_delete = np.setdiff1d(np.array(list(word_counts_by_document[k].keys())), np.array(list(v.keys())))
        for k2,v2 in v.items():
            n += word_counts_by_document[k][k2]
            vocab_new.add(k2)
        for k2 in to_delete:
            del word_counts_by_document[k][k2]
    return word_counts_by_document, vocab_new, n

def convert_bow(bow):
    return {d:np.hstack([[k]*v for k,v in bag.items()]) for d,bag in enumerate(bow)} # convert the dictionary of word counts into an array