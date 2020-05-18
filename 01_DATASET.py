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
import os

#abre a camera, setar 0 para camera embutida ou usar seu IP do DROIDCAM (apk android) caso nao tenha webcam
camera = cv2.VideoCapture(0)
#classificador haarcascade
classificador = cv2.CascadeClassifier('trainer/haarcascade_frontalface_default.xml')

# Para cadastrar cada pessoa vamos entrar com uma ID
face_id = input('\n Digite uma ID [1-4]:  ')
print('\n [INFO] Inicializando a captura de imagens, olhe fixamente para camera ...')

# Contador para quantidade de fotos que vamos ter salvas no dataset
contador_fotos = 0

#loop essencial da camera
while True:
    #le a camera
    retorno, imagem = camera.read()
    #converte a imagem para cinza pois o opencv classifica melhor
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    #aplica o classificador na imagem
    faces = classificador.detectMultiScale(cinza,scaleFactor =1.1, minNeighbors = 8,  minSize = (60,60))
    #loop para criar os quadrados de reconhecimento
    for (x,y,w,h) in faces:
        #cria um retangulo em volta da nossa imagem para marcar o reconhecimento
        cv2.rectangle(imagem, (x, y), (x + w, y + h), (255, 0, 0), 2)
        contador_fotos += 1

        # Salva as imagens capturadas na pasta dataset
        cv2.imwrite('dataset/face.' + str(face_id) + '.' + str(contador_fotos) + '.jpg', cinza[y:y + h, x:x + w])

    # Presione ESC para sair
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    # Pega 30 imagens para o contador
    elif contador_fotos >= 30:
         break

#Final que fecha o codigo
print('\n [INFO] Todas imagens foram salvas na pasta dataset/, execute o trenamento')
camera.release()
cv2.destroyAllWindows()


