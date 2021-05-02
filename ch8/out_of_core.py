'''アウトオブコア学習'''
import re
import numpy as np
import pyprind
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version


stop = stopwords.words('english')

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        # ヘッダーを読み飛ばす
        next(csv)
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
            return None, None
    return docs, y

vect = HashingVectorizer(decode_error='ignore', n_features=2**21,
    preprocessor=None, tokenizer=tokenizer)

clf = SGDClassifier(loss='log', random_state=1)
doc_stream = stream_docs(path='/Users/rukaoide/\
Library/Mobile Documents/com~apple~CloudDocs/\
Documents/Python/12_machine_learning_book3/ch8/movie_data.csv')

pbar = pyprind.ProgBar(45)

# ミニバッチで学習(45000個の文書)
classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()

# モデルの性能を評価(5000個の文書)
X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))
