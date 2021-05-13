'''感情分析'''
import re
import pandas as pd
import numpy as np
import nltk
import pyprind
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


'''データ前処理'''
df = pd.read_csv('/Users/rukaoide/Library/Mobile Documents/\
com~apple~CloudDocs/Documents/Python/12_machine_learning_book3/\
ch8/movie_data.csv', encoding='utf-8')
df.head()
df.shape

#%% 単語を特徴量ベクトルに変換する
count = CountVectorizer()
docs = np.array([
    'The sun is shining',
    'The weather is sweet',
    'The sun is shining, the weather is sweet, and one and one is two'])
bag = count.fit_transform(docs)
# 単語と整数の対応
print(count.vocabulary_)
# 特徴量ベクトルの表示
print(bag.toarray())

#%% TF-IDFを使って単語の関連性を評価
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
# 表示桁数を指定
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

#%% テキストデータのクレンジング
# 意味のない文字が多い
df.loc[0, 'review'][-50:]

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
    ' '.join(emoticons).replace('-', ''))
    return text

preprocessor(df.loc[0, 'review'][-50:])

# 全ての映画レビューにpreprocessorを適用
df['review'] = df['review'].apply(preprocessor)

#%% 文書をトークン化する
# 空白文字で区切る
def tokenizer(text):
    return text.split()
# 例
tokenizer('runners like running and thus they run')

# ワードステミング
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
# 例
tokenizer_porter('runners like running and thus they run')

# ストップワードの除去
nltk.download('stopwords')
stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:]
    if w not in stop]


'''文書解析'''
#%% ロジスティック回帰モデルの訓練
# 25000個の訓練用文書とテスト用文書に分ける
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

# 5分割交差検証
tfidf = TfidfVectorizer(strip_accents=None,
    lowercase=False, preprocessor=None)
param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},]
lr_tfidf = Pipeline([('vect', tfidf),
    ('clf', LogisticRegression(random_state=0, solver='liblinear'))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy',
    cv=5, verbose=2, n_jobs=-1)
gs_lr_tfidf.fit(X_train, y_train)

pbar = pyprind.ProgBar(50000)

print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
# テスト
clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))
