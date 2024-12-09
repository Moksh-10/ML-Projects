import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub

df=pd.read_csv("wine-reviews-1.csv" , usecols = ['country' , 'description' , 'points' , 'price' , 'variety' , 'winery'])

df.head()

df = df.dropna(subset=["description","points"])

plt.hist(df.points , bins=20)
plt.title("Points histogram")
plt.ylabel("N")
plt.xlabel("Points")
plt.show()

df["label"]=(df.points>=90).astype(int)
df=df[["description","label"]]

train,val,test=np.split(df.sample(frac=1),[int(0.8*len(df)),int(0.9*len(df))])

len(train),len(val),len(test)

def df_to_dataset(dataframe,shuffle=True,batch_size=1024):
  df=dataframe.copy()
  labels=df.pop('label')
  df=df["description"]
  ds=tf.data.Dataset.from_tensor_slices((df,labels))
  if shuffle:
    ds=ds.shuffle(buffer_size=len(dataframe))
  ds=ds.batch(batch_size)
  ds=ds.prefetch(tf.data.AUTOTUNE)
  return ds

train_data=df_to_dataset(train)
valid_data=df_to_dataset(val)
test_data=df_to_dataset(test)

embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer=hub.KerasLayer(embedding, dtype=tf.string, trainable=True)

hub_layer(list(train_data)[0][0])

model=tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16,activation='relu'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(16,activation='relu'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

model.evaluate(train_data)

model.evaluate(valid_data)

history=model.fit(train_data,epochs=5,validation_data=valid_data)

plt.plot(history.history['accuracy'],label="Training acc")
plt.plot(history.history['val_accuracy'],label="Validation acc")
plt.title("Accuracy of model")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.show()

plt.plot(history.history['loss'],label="Training loss")
plt.plot(history.history['val_loss'],label="Validation loss")
plt.title("Loss of model")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.show()

model.evaluate(test_data)

encoder=tf.keras.layers.TextVectorization(max_tokens=2000)
encoder.adapt(train_data.map(lambda text , label:text))

vocab=np.array(encoder.get_vocabulary())
vocab[:20]

model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=32,
        mask_zero=True
    ),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

model.evaluate(train_data)
model.evaluate(valid_data)

history=model.fit(train_data,epochs=5,validation_data=valid_data)

model.evaluate(test_data)