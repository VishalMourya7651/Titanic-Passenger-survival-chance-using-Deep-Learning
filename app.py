import streamlit as st
import pandas as pd
import tensorflow
from tensorflow.keras.models import load_model
import pickle

st.title('Titanic Survival Prediction of a Passenger')

pclass=st.slider('Enter the passenger class',1,3)

sex=st.selectbox('Select the passenger gender',['male','female'])
sibsp=st.slider('Enter the number of siblings and spouses aboard',1,8)

parch=st.slider('Enter the number of parents and children aboard',0,6)
fare=st.number_input('Enter the fare paid by the passenger')

embarked=st.selectbox('Select the port of embarkation',['Southampton','Cherbourg','Queenstown'])

data=pd.DataFrame([{'Pclass':pclass,'Sex':sex,'SibSp':sibsp,'Parch':parch,'Fare':fare,'Embarked':embarked}])

if st.button('Data'):
    st.write(data)

model=load_model('titanic_model.h5')

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

with open('onehot_encoder.pkl','rb') as file:
    onehot =pickle.load(file)

with open('label_encoder.pkl','rb') as file:
    label =pickle.load(file) 

data['Sex']=label.transform(data['Sex'])

embarked=onehot.transform(data[['Embarked']])

embarked=pd.DataFrame(embarked,columns=onehot.get_feature_names_out([['Embarked']]))
data=pd.concat([data.drop(columns=['Embarked']), embarked], axis=1)

data[['Pclass','SibSp','Parch','Fare']]=scaler.transform(data[['Pclass','SibSp','Parch','Fare']])

y=model.predict(data)
y=y[0][0]


def Chance(y):
    if y>0.5:
        return 'The passenger is likely to survive.'
    else:
        return 'The passenger is likely to not survive.'

    
if st.button('Predict survival chance'):
    st.write(f'The predicted survival probability is {y:.2f}.')
    st.write(Chance(y))
