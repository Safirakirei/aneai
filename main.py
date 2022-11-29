#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st 

#read the data
dataset = pd.read_csv("anemia.csv")

#shape of the data
print(dataset.shape)

#first row of the data
print(dataset.head())

#checking missing values in the data
print(dataset.isnull().sum())

#separating the independent and dependent variable
y = dataset['Result']
X = dataset.drop(['Result'], axis = 1)

#import train_test_split to create validation set
from sklearn.model_selection import train_test_split

#creating the train and validation set
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state = 101, stratify = y, test_size = 0.25)

#distribution in training set (berapa banyak true-false target di data training)
print(y_train.value_counts(normalize = True))

#distribution in validation set (berapa banyak true-false target di data test (validation))
print(y_valid.value_counts(normalize = True))

#See the shape training and validation
print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)

#import decision tree classifier
from sklearn.tree import DecisionTreeClassifier

#creating the decision tree function
dt_model = DecisionTreeClassifier(criterion = 'entropy')

#fitting the model
dt_model.fit(X_train, y_train)

#check the training score (Seberapa valid (bener ngga) sama data training)
print(dt_model.score(X_train, y_train))

#check the validation score (Seberapa valid (bener ngga) sama data test)
print(dt_model.score(X_valid, y_valid))

#Predictions on validation set (check hasil result dari input (data) yg di test)
#print(dt_model.predict(X_valid))
dt_model.predict(X_valid)

#cuma nyoba, di dataset kita ngga ngaruh si kayanya
y_pred = dt_model.predict_proba(X_valid)[:,1]
new_y = []
for i in range(len(y_pred)):
    if y_pred[i] < 0.6:
        new_y.append(0)
    else:
        new_y.append(1)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_valid, new_y))

#ini juga nyoba diganti max_dept nya buat nyari accuracy tertinggi
train_accuracy = []
validation_accuracy = []
for depth in range(1,10):
    dt_model = DecisionTreeClassifier(criterion = 'entropy', max_depth = depth)
    dt_model.fit(X_train, y_train)
    train_accuracy.append(dt_model.score(X_train, y_train))
    validation_accuracy.append(dt_model.score(X_valid, y_valid))
frame = pd.DataFrame({'max_depth': range(1,10), 'train_acc': train_accuracy, 'valid_acc': validation_accuracy})
print(frame.head())
plt.figure(figsize=(12,6))
plt.plot(frame['max_depth'], frame['train_acc'], marker='o')
plt.plot(frame['max_depth'], frame['valid_acc'], marker='o')
plt.xlabel('Depth of tree')
plt.ylabel('performance')
# plt.legend() #ini ngga bisa ntah kenapa wkwk
# plt.show()
#Dari hasil plot didapetin acc_valid tertinggi di 3 keatas, jadi diambil max_depth = 3

train_accuracy = []
validation_accuracy = []
for leaf in range(2,10):
    dt_model = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, max_leaf_nodes = leaf)
    dt_model.fit(X_train, y_train)
    train_accuracy.append(dt_model.score(X_train, y_train))
    validation_accuracy.append(dt_model.score(X_valid, y_valid))
frame = pd.DataFrame({'max_depth': range(2,10), 'train_acc': train_accuracy, 'valid_acc': validation_accuracy})
print(frame.head())
plt.figure(figsize=(12,6))
plt.plot(frame['max_depth'], frame['train_acc'], marker='o')
plt.plot(frame['max_depth'], frame['valid_acc'], marker='o')
plt.xlabel('Depth of tree')
plt.ylabel('Ammount of leaf node')
# plt.legend() #ini ngga bisa ntah kenapa wkwk
# plt.show()
#Dari hasil plot didapetin acc_vavlid tertinggi kalo leaf node nya 4 ke atas, jadi diambil max_leaf_node = 4

#decision tree akhir dengan parameter yg td di cek
dt_model = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, max_leaf_nodes = 4)

#fitting the model
dt_model.fit(X_train, y_train)

#training score
print(dt_model.score(X_train, y_train))

#validation score
print(dt_model.score(X_valid, y_valid))

from sklearn import tree

# decision_tree = tree.export_graphviz(dt_model, out_file='nyenye.dot', feature_names = X_train.columns, filled = True)

st.write("Dengan Menggunakan Decision Tree Nilai Akurasinya Adalah:")
st.write(accuracy_score(y_test, y_pred_dt))

st.write("# Masukkan data")


form = st.form(key='my-form')
inputGender = form.number_input("Jenis kelamin (1 = male, 0 = female): ", 0)
inputAge = form.number_input("Umur: ", 0)
inputHyper = form.number_input("Apakah mempunyai hipertensi? (1 = ya, 0 = tidak): ", 0)
inputHD = form.number_input("Apakah mempunyai penyakit jantung? (1 = ya, 0 = tidak): ", 0)
inputGlucose = form.number_input("Rata-rata kadar glukosa: ", 0)
inputBMI = form.number_input("BMI: ", 0)
submit = form.form_submit_button('Submit')

completeData = np.array([inputGender, inputAge, inputHyper, 
                        inputHD, inputGlucose, inputBMI]).reshape(1, -1)
scaledData = ss_train_test.transform(completeData)


st.write('Tekan Submit Untuk Melihat Hasil Prediksi')

if submit:
    prediction = m.predict(scaledData)
    if prediction == 1 :
        result = 'Stroke'
    else:
        result = 'Sehat'
    st.write(result)