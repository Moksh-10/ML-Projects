import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


df=pd.read_csv("diabetes.csv")
df.head()

for i in range(len(df.columns[:-1])):
  label=df.columns[i]
  plt.hist(df[df['Outcome']==1][label],color='blue',label="Diabetes",alpha=0.7,density=True,bins=15)
  plt.hist(df[df['Outcome']==0][label],color='red',label="No Diabetes",alpha=0.7,density=True,bins=15)
  plt.title(label)
  plt.ylabel("Probability")
  plt.xlabel(label)
  plt.legend()
  plt.show()

X=df[df.columns[:-1]].values
Y=df[df.columns[-1]].values

scaler = StandardScaler()
X=scaler.fit_transform(X)
data=np.hstack((X,np.reshape(Y,(-1,1))))
tranformed_df=pd.DataFrame(data,columns=df.columns)

over = RandomOverSampler()
X,Y=over.fit_resample(X,Y)
data=np.hstack((X,np.reshape(Y,(-1,1))))
tranformed_df=pd.DataFrame(data,columns=df.columns)

len(tranformed_df[tranformed_df["Outcome"]==1]) , len(tranformed_df[tranformed_df["Outcome"]==0])

X_train , X_temp , Y_train , Y_temp = train_test_split(X,Y,test_size=0.4,random_state=0)
X_valid , X_test , Y_valid , Y_test = train_test_split(X_temp,Y_temp,test_size=0.5,random_state=0)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

model.evaluate(X_train , Y_train)

model.evaluate(X_valid,Y_valid)

model.fit(X_train,Y_train,batch_size=16,epochs=20,validation_data=(X_valid,Y_valid))

model.evaluate(X_test,Y_test)

