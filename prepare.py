#/**
#* Copyright (c) 2019, Vsevolod Averkov <averkov@cs.petrsu.ru>
#*
#* This code is licensed under a MIT-style license.
#*/
import keras.models
import random 
import tensorflow
from PIL import Image, ImageDraw,ImageFont
i=1
z= ImageFont.truetype('FreeMono.ttf', 100)

for  i  in range(420):
    s=str(i)
    img = Image.new('RGB', (100, 100), color = (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((random.randint(1,80),random.randint(1,50)), "0", fill=(0,0,0),font=z)
    
    img.save('zeroes/'+'zero' + s +'.jpg')

i=1

for  i  in range(420):
    s=str(i)
    img = Image.new('RGB', (100, 100), color = (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((random.randint(1,80),random.randint(1,50)), "1", fill=(0,0,0),font=z)
    
    img.save('one/'+"one" +s +'.jpg')


i=1
for  i  in range(420):
    s=str(i)
    
    img = Image.new('RGB', (100, 100), color = (255, 255, 255))
    d = ImageDraw.Draw(img)
    
    d.text((random.randint(1,80),random.randint(1,50)), "3", fill=(0,0,0),font=z)
    img.save('three/'+ "three" + s +'.jpg')    


i=1
for  i  in range(420):
    s=str(i)
    img = Image.new('RGB', (100, 100), color = (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((random.randint(1,80),random.randint(1,50)), "8", fill=(0,0,0),font=z)
    
    img.save('eight/'+"eight"+s +'.jpg')    