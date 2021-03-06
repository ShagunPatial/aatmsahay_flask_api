# -*- coding: utf-8 -*-
"""Aatmsahay ML model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17BlcoHEUhrBHSZNKSiHj7n4xrgObBaCL
"""

import pandas as pd
import numpy as np

df = pd.read_excel('DATASET.xlsx',engine='openpyxl')

# df.drop_duplicates(inplace = True)

df['Disease'].value_counts()

df.isna().sum()

df['Symptom_1'] = df['Symptom_1'].astype(str)
df['Symptom_2'] = df['Symptom_2'].astype(str)
df['Symptom_3'] = df['Symptom_3'].astype(str)
df['Symptom_4'] = df['Symptom_4'].astype(str)
df['Symptom_5'] = df['Symptom_5'].astype(str)
df['Symptom_6'] = df['Symptom_6'].astype(str)
df['Symptom_7'] = df['Symptom_7'].astype(str)
df['Symptom_8'] = df['Symptom_8'].astype(str)
df['Symptom_9'] = df['Symptom_9'].astype(str)
df['Symptom_10'] = df['Symptom_10'].astype(str)
df['Symptom_11'] = df['Symptom_11'].astype(str)
df['Symptom_12'] = df['Symptom_12'].astype(str)
df['Symptom_13'] = df['Symptom_13'].astype(str)
df['Symptom_14'] = df['Symptom_14'].astype(str)
df['Symptom_15'] = df['Symptom_15'].astype(str)
df['Symptom_16'] = df['Symptom_16'].astype(str)
df['Symptom_17'] = df['Symptom_17'].astype(str)

df['Combined'] = df[['Symptom_1', 'Symptom_2','Symptom_3','Symptom_4','Symptom_5','Symptom_6','Symptom_7','Symptom_8','Symptom_9','Symptom_10','Symptom_11','Symptom_12','Symptom_13','Symptom_14','Symptom_15','Symptom_16','Symptom_17']].apply(lambda x: ' '.join(x), axis=1)

df['Combined'][0]

df = df.reset_index()

del df['index']

corpus = []
for i in range(0,len(df)):
    words = df['Combined'][i]
    words = words.replace('nan','')
    words = words.rstrip()
    corpus.append(words)

df['Cleared Combined'] = corpus

df.shape

df.head()

x = df['Cleared Combined']

y = df['Disease']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state = 7, stratify = y)

x_train.shape, x_test.shape, y_train.shape, y_test.shape

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

from sklearn.feature_extraction.text import TfidfVectorizer
tfvec= TfidfVectorizer()

x_train = tfvec.fit_transform(x_train).toarray()
x_test = tfvec.transform(x_test).toarray()

x_train[0]

from sklearn.ensemble import RandomForestClassifier

disease_detect_rfc = RandomForestClassifier().fit(x_train,y_train)

y_pred_rfc = disease_detect_rfc.predict(x_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_pred_rfc))

from sklearn.naive_bayes import MultinomialNB

disease_detect_nb = MultinomialNB().fit(x_train,y_train)

y_pred_nb = disease_detect_nb.predict(x_test)

print(accuracy_score(y_test,y_pred_nb))

from sklearn.neighbors import KNeighborsClassifier

disease_detect_knn = KNeighborsClassifier().fit(x_train,y_train)

y_pred_knn = disease_detect_knn.predict(x_test)

print(accuracy_score(y_test,y_pred_knn))

# a = tfvec.transform([str(input("Enter the symptoms you're feeling: "))]).toarray()

# le.inverse_transform([disease_detect_knn.predict(a),disease_detect_nb.predict(a),disease_detect_rfc.predict(a)])

import pickle

# pickling the machine learning models
pickle.dump(disease_detect_rfc, open('rdf_model.pkl','wb'))
pickle.dump(disease_detect_nb, open('nb_model.pkl','wb'))
pickle.dump(disease_detect_knn, open('knn_model.pkl','wb'))

# model = pickle.load(open('rdf_model.pkl','rb'))
# print(le.inverse_transform(model.predict(tfvec.transform([str("joint_pain vomiting fatigue")]).toarray())))

# pickling the encoders
with open('le.pkl', 'wb') as out_tfidf:
    pickle.dump(le, out_tfidf)
with open('tfvec.pkl', 'wb') as out_le:
    pickle.dump(tfvec, out_le)