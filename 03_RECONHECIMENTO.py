#!/usr/bin/env python
# -*- coding: utf-8 -*-
#███╗   ███╗ █████╗ ███╗   ██╗██╗ ██████╗ ██████╗ ███╗   ███╗██╗ ██████╗
#████╗ ████║██╔══██╗████╗  ██║██║██╔════╝██╔═══██╗████╗ ████║██║██╔═══██╗
#██╔████╔██║███████║██╔██╗ ██║██║██║     ██║   ██║██╔████╔██║██║██║   ██║
#██║╚██╔╝██║██╔══██║██║╚██╗██║██║██║     ██║   ██║██║╚██╔╝██║██║██║   ██║
#██║ ╚═╝ ██║██║  ██║██║ ╚████║██║╚██████╗╚██████╔╝██║ ╚═╝ ██║██║╚██████╔╝
#╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝ ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝ ╚═════╝
#            @GorpoOrko | Manicomio TCXS Project | 2020
import cv2
import numpy as np
from PIL import Image
import os
import cv2




#classificador, reconhecedor e caminho do trainer
reconhecedor = cv2.face.LBPHFaceRecognizer_create()
reconhecedor.read('trainer/trainer.yml')
classificador = cv2.CascadeClassifier('trainer/haarcascade_frontalface_default.xml')

#contador das ID's
contador_id = 0
#nomes relacionados as ID's: 'Gorpo' = 1, 'Eddie' = 2, 'slipknot' = 3 .....
nomes = ['None',  'Beavis',  'Butt-Head', 'jose', 'maria']

# Inicializa a camera
camera = cv2.VideoCapture(0)   #'http://192.168.0.4:4747/mjpegfeed'


while True:
    # le a camera e converte pra cinza
    ret, img =camera.read()
    #converte para cinza
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # aplica o classificador na imagem
    faces = classificador.detectMultiScale(gray, scaleFactor =1.1, minNeighbors = 8,  minSize = (60,60))
    #faces = classificador.detectMultiScale(gray, scaleFactor =1.08, minNeighbors = 4,  minSize = (30,30))

    # loop para criar os quadrados de reconhecimento
    for(x,y,w,h) in faces:
        # cria um retangulo em volta da nossa imagem para marcar o reconhecimento
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 0)
        contador_id, confianca = reconhecedor.predict(gray[y:y + h, x:x + w])

        # Verifique se a confiança é menor que 100 ==> "0" é a combinação perfeita
        if (confianca < 100):
            contador_id = nomes[contador_id]
            confianca = "  {0}%".format(round(100 - confianca))
        else:
            contador_id = "desconhecido"
            confianca = "  {0}%".format(round(100 - confianca))

        #exibe os textos na caixa do retangulo
        cv2.putText(img, str(contador_id), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0), 2, cv2.LINE_AA)
        cv2.putText(img, str(confianca), (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0), 4, cv2.LINE_AA)

    #Final do codigo se clicado ESC sai do loop e encerra a camera
    cv2.imshow('camera',img)
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

camera.release()
cv2.destroyAllWindows()
