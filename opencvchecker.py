#/**
#* Copyright (c) 2019, Vsevolod Averkov <averkov@cs.petrsu.ru>
#*
#* This code is licensed under a MIT-style license.
#*/
#python opencvchecker.py --image test/zero/zero55.jpg --model lab.model --label-bin  lab.pickle --width 32 --height 32 --flatten 1
#                                                                        --width 64 --height 64
from keras.models import load_model
import argparse
import pickle
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="КАРТИНКА")
ap.add_argument("-m", "--model", required=True,
	help="МОДЕЛЬ KERAS")
ap.add_argument("-l", "--label-bin", required=True,
	help="Заголовки")
ap.add_argument("-w", "--width", type=int, default=28,
	help="целевая ширина пространственного измерения")
ap.add_argument("-e", "--height", type=int, default=28,
	help="высота")
ap.add_argument("-f", "--flatten", type=int, default=-1,
	help="сглаживание")
args = vars(ap.parse_args())

# загрузить входное изображение и изменить его размер до целевых пространственных размеров
image = cv2.imread(args["image"])

output = image.copy()
image = cv2.resize(image, (args["width"], args["height"]))

# масштабировать значения пикселей до [0, 1]
image = image.astype("float") / 255.0

#  должны ли мы сгладить изображение 
# 
if args["flatten"] > 0:
	image = image.flatten()
	image = image.reshape((1, image.shape[0]))

#в противном случае  - не сглаживать
# изображение
else:
	image = image.reshape((1, image.shape[0], image.shape[1],
		image.shape[2]))

# загрузить модель
print("[INFO] Загружаю модель")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())

preds = model.predict(image)

#находим метки класса с наибольшей вероятностью
i = preds.argmax(axis=1)[0]
label = lb.classes_[i]

# рисуем итог
text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
cv2.putText(output, text, (1, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
	(0, 255, 0), 2)

# показываем
cv2.imshow("Image", output)
cv2.waitKey(0)