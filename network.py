#/**
#* Copyright (c) 2019, Vsevolod Averkov <averkov@cs.petrsu.ru>
#*
#* This code is licensed under a MIT-style license.
#*/
# python lab.py -d train -m lab.model -l lab.pickle -p plot.png
import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
from keras.models import load_model
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="img")
ap.add_argument("-m", "--model", required=True,
	help="model")
ap.add_argument("-l", "--label-bin", required=True,
	help="label ")
ap.add_argument("-p", "--plot", required=True,
	help="picture")
args = vars(ap.parse_args())

# инициализируем
print("[INFO] Загружаю картинки")
data = []
labels = []

# захватить пути изображения и случайным образом перемешать их
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(200)
random.shuffle(imagePaths)


for imagePath in imagePaths:
	# загрузить изображение, изменить его размер до 32x32 пикселей (игнорируя
	#(соотношение сторон),  изображение в 32x32x3 = 3072 пикселей изображения
	
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (32, 32)).flatten()
	data.append(image)

	# извлечь метку класса из пути к изображению и обновить
	# список меток
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

# масштабировать  интенсивности пикселей до диапазона [0, 1
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)


# разбить данные на разделы обучения и тестирования, используя 75%
# данные для обучения и оставшиеся 25% для тестирования
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=200)

# конвертировать метки из целых чисел в векторы 
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

#Keras
def create_model():
		model = Sequential()
		
		model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
		model.add(Dense(512, activation="sigmoid"))
		model.add(Dense(len(lb.classes_), activation="softmax"))
		return model



model=create_model()
model.load_weights("weights.h5")
INIT_LR = 0.01
EPOCHS = 20
print("[INFO] тренерую")
opt = SGD(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# тренеровка
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=EPOCHS, batch_size=32)


predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))


score = model.evaluate(testX, testY, verbose=0)
print("Точность: %.2f%%" % (score[1]*100))

# графики
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Тренировка нейросети  Loss and Accuracy")
plt.xlabel("Эпохи")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])

print("[INFO] сохраняю")
model.save_weights("weights.h5")
model.save(args["model"])
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()