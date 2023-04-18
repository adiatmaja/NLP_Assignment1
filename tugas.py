import numpy as np
import pandas as pd
import os
from tkinter import Tcl
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

import re
import string

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from nltk.corpus import words

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')

english_words = set(words.words())
stemmer = PorterStemmer()
vectorizer = TfidfVectorizer()

stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

stop_words_english = set(stopwords.words('english'))
stop_words_indonesia = set(stopwords.words('indonesian'))

def preprocess(content, language):
    pattern = r"\d+|[" + re.escape(string.punctuation) + "]|\[[^]]*\]|\([^)]*\)|\w*-\w*"
    pattern = re.sub(pattern, '', content)
    pattern = pattern.lower()
    if(language == 'inggris'):
        words_list = nltk.word_tokenize(pattern)
        words = [word for word in words_list if word in english_words and not word in stop_words_english and len(word) > 1]
        words = " ".join(words)
    else:
        tokens = stopword_remover.remove(pattern).split()
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        words = " ".join(stemmed_tokens)
        words = " ".join(w for w in words.split() if w not in stop_words_indonesia)
        if words not in stop_words_indonesia:
            words = words
    return words

def process_files(language, root):
    texts = []
    try:
        dir_path = os.path.join(root, language)
        # Cek apakah direktori ada atau tidak
        # Menghindari jika terdapat 'hidden file' yang tidak bisa di akses
        if not os.path.exists(dir_path):
            print(f"Directory {dir_path} does not exist")
            return []
        
        files = sorted(os.listdir(dir_path))
        for txt in files:
            file_path = os.path.join(root, language, txt)
            with open(file_path, 'r', encoding="utf8") as f:
                content = f.read()
                words = preprocess(content, language)
                texts.append(words)
    except Exception as e:
        print(f"Error processing files in directory {dir_path}: {e}")
    return texts

def count_tf_idf_indo(texts):
    tfidf_matrix = vectorizer.fit_transform(texts[0:5])
    dense_matrix = tfidf_matrix.todense()
    feature_names = sorted(vectorizer.vocabulary_.keys())
    doc_names = ["Dokumen" + str(i+1) for i in range(len(texts[0:5]))]
    df = pd.DataFrame(dense_matrix, columns=feature_names, index=doc_names)
    df_transposed_indo = df.transpose()
    
    terms = vectorizer.get_feature_names_out()

    # sum tfidf frequency of each term through documents
    sums = tfidf_matrix.sum(axis=0)

    # connecting term to its sums frequency
    data = []
    for col, term in enumerate(terms):
        data.append( (term, sums[0,col] ))

    ranking = pd.DataFrame(data, columns=['term','rank'])
    print('Ranking term pada Dokumen TF-IDF Bahasa Indonesia : ')
    print(ranking.sort_values('rank', ascending=False))

    print('Dokumen TF-IDF Bahasa Indonesia : ')
    return df_transposed_indo

def count_tf_idf_inggris(texts):
    tfidf_matrix = vectorizer.fit_transform(texts[5:])
    dense_matrix = tfidf_matrix.todense()
    feature_names = sorted(vectorizer.vocabulary_.keys())
    doc_names = ["Dokumen" + str(i+1) for i in range(len(texts[5:]))]
    df = pd.DataFrame(dense_matrix, columns=feature_names, index=doc_names)
    df_transposed = df.transpose()

    terms = vectorizer.get_feature_names_out()

    # sum tfidf frequency of each term through documents
    sums = tfidf_matrix.sum(axis=0)

    # connecting term to its sums frequency
    data = []
    for col, term in enumerate(terms):
        data.append( (term, sums[0,col] ))

    ranking = pd.DataFrame(data, columns=['term','rank'])
    print('Ranking term pada Dokumen TF-IDF Bahasa Inggris : ')
    print(ranking.sort_values('rank', ascending=False))

    print('Dokumen TF-IDF Bahasa Inggris : ')
    return df_transposed

def main():
    root = 'text'
    listFolder = sorted(os.listdir(root))
    texts = []
    for language in listFolder:
        texts += process_files(language, root)

    # Membuat file Indonesia.csv untuk menampung hasil tf-idf
    df_transposed_indo = count_tf_idf_indo(texts)
    print(df_transposed_indo)
    df_transposed_indo.to_csv('Indonesia.csv', index=True)

    # Membuat file Indonesia.csv untuk menampung hasil tf-idf
    df_transposed_inggris = count_tf_idf_inggris(texts)
    print(df_transposed_inggris)
    df_transposed_inggris.to_csv('Inggris.csv', index=True)

if __name__ == '__main__':
    texts = main()