import pandas as pd
import warnings
import xgboost as xgb
import nltk
import re
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from config import Config as cfg
warnings.simplefilter('ignore')


def stem_words(text, stemmer):
    tokens = re.findall('([a-zA-Zа-яА-Я]{3,})', text)
    tokens = [i for i in tokens if not(i in nltk.corpus.stopwords.words('russian'))]
    tokens = ' '.join([stemmer.stem(token) for token in tokens])
    return(tokens)


def fit_xgb(train, label, cols):
    dtrain = xgb.DMatrix(train[cols], label=train[label])
    model = xgb.train(cfg.param, dtrain, cfg.numround[label])
    return(model)


def predict_xgb(train, test, label, cols):
    model = fit_xgb(train, label, cols)
    dtest = xgb.DMatrix(test[cols])
    pred = model.predict(dtest)
    return(pred)


def load_data():
    train = pd.read_csv(cfg.data_dir + cfg.train_file, sep='\t')
    test = pd.read_csv(cfg.data_dir + cfg.test_file, sep='\t')
    return(train, test)


def process_text(train, test):
    train['text'] = [x.lower() for x in train['text']]
    test['text'] = [x.lower() for x in test['text']]

    stemmer = SnowballStemmer("russian")

    train['text_stem'] = [stem_words(train['text'][i], stemmer) for i in range(train.shape[0])]
    test['text_stem'] = [stem_words(test['text'][i], stemmer) for i in range(test.shape[0])]
    return(train, test)


def make_tfidf_features(train, test):
    vectorizer = TfidfVectorizer(max_features=cfg.tfidf_features, ngram_range=(1, 2))

    tfidf_cols = ['tfidf_stem_'+str(i) for i in range(cfg.tfidf_features)]

    tfidf = vectorizer.fit_transform(train['text_stem'])
    tfidf = pd.DataFrame(tfidf.todense(), columns=tfidf_cols)

    train = pd.concat([train, tfidf], axis=1)

    tfidf = vectorizer.transform(test['text_stem'])
    tfidf = pd.DataFrame(tfidf.todense(), columns=tfidf_cols)

    test = pd.concat([test, tfidf], axis=1)
    return(train, test, tfidf_cols)


def get_predictions(train, test, tfidf_cols):
    predictions = {}
    for label in cfg.labels:
        predictions[label] = predict_xgb(train, test, label, tfidf_cols)

    predictions = pd.DataFrame.from_dict(predictions)
    return(predictions)


def write_predictions(predictions):
    predictions.to_csv(cfg.data_dir+cfg.output_prob_file, index=False, sep=';')
    (predictions > cfg.threshold).astype('int').to_csv(cfg.data_dir+cfg.output_label_file, index=False, sep=';')
