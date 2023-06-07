import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

(X_train,y_train) , (X_test , y_test) = keras.datasets.mnist.load_data()

print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))

print(X_train[0].shape)
# MNIST veri kümesindeki her örnek, 28x28 piksel boyutunda siyah beyaz bir görüntüdür. Bu nedenle, X_train[0].shape ifadesi, 
# eğitim veri setindeki ilk örneğin boyutunu döndürür. 
# Eğer X_train veri kümesinde 60.000 örnek varsa, X_train[0].shape ifadesi (28, 28) şeklinde bir çıktı üretecektir.

X_train = X_train / 255
X_test = X_test / 255
print(X_train[0])
# bazı makine öğrenimi veya derin öğrenme algoritmaları, girdi verilerinin 0 ile 1 arasında veya 
# -1 ile 1 arasında ölçeklendirilmesini tercih eder.
# Bu nedenle,X_train = X_train / 255 ifadesi, eğitim veri setindeki piksel değerlerini 255'e böler ve böylece her piksel değerinin
# 0 ile 1 arasında olmasını sağlar. Aynı şekilde, X_test = X_test / 255 ifadesi de test veri setindeki piksel değerlerini
# 255'e böler.



print(plt.matshow(X_train[0]))
print(y_train[0])
print(y_train[2])
print(y_train[:10])

X_train_yenidenboyutlama = X_train.reshape(len(X_train),28*28)
print(X_train_yenidenboyutlama.shape) #(60000, 784)
print(X_train_yenidenboyutlama[0].shape) # (784,)
# X_train_yenidenboyutlama = X_train.reshape(len(X_train), 28*28) ifadesi,
# MNIST veri kümesindeki eğitim veri setinin boyutunu yeniden şekillendirerek her görüntüyü tek bir vektör olarak düzenlemek
# için kullanılan bir Python kodudur.

X_test_yenidenboyutlama = X_test.reshape(len(X_test),28*28)
print(X_test_yenidenboyutlama.shape) #(60000, 784)
print(X_test_yenidenboyutlama[0].shape) #(784,)




model = keras.Sequential([
    keras.layers.Dense(10, input_shape = (784,),activation='sigmoid')
])

model.compile(optimizer='adam',loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train_yenidenboyutlama , y_train , epochs = 5)

print(model.evaluate(X_test_yenidenboyutlama, y_test))


print(plt.matshow(X_test[0])) # 7 veriyor
y_predict  = model.predict(X_test_yenidenboyutlama)
print(np.argmax(y_predict[0])) # 7 veriyor
# X_TEST DEĞERLERİYLE TAHMİN DEĞERLERİ ÖRTÜŞÜYOR DEMEK.



y_predicted_labels = [np.argmax(i) for i in y_predict]
print(y_predicted_labels[:5])

cm = tf.math.confusion_matrix(labels = y_test , predictions=y_predicted_labels)
print(cm)




# USING HIDDEN LAYER (Gizli katmanı kullanma)

model = keras.Sequential([
    keras.layers.Dense(100 , input_shape = (784,),activation='relu'),
    keras.layers.Dense(10 , activation = 'sigmoid')
])

model.compile(optimizer='adam',loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train_yenidenboyutlama,y_train,epochs=5)


print(model.evaluate(X_test_yenidenboyutlama,y_test))

y_predicted = model.predict(X_test_yenidenboyutlama)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
print(y_predicted_labels[:5])

cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
print(cm)


# Using Flatten layer so that we don't have to call .reshape on input dataset
# Giriş veri kümesinde .reshape'i çağırmak zorunda kalmamak için Düzleştir katmanını kullanma


model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(100 , activation='relu'),
    keras.layers.Dense(10 , activation='sigmoid')

])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)

print(model.evaluate(X_test,y_test))
