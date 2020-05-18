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

#classificador, reconhecedor e caminho da dataset
classificador = cv2.CascadeClassifier("trainer/haarcascade_frontalface_default.xml");
reconhecedor = cv2.face.LBPHFaceRecognizer_create()
caminho_dataset = 'dataset'


# função para pegar as imagens e a ID fornecida no script anterior
def imagensIds(caminho_dataset):
    #pega o caminho do dataset e faz um loop para reconhecer todas imagens e grava nesta lista
    caminho_dataset = [os.path.join(caminho_dataset, f) for f in os.listdir(caminho_dataset)]
    #cria uma lista com todos os rostos encontrados na pasta dataset
    dataset_faces=[]
    #cria uma lista com todas id's que estão embutidas na imagem, por isto usar de 1-4
    ids = []

    #pega todas imagens
    for caminho_imagem in caminho_dataset:
        #abre a imagem com o PIL em escala de cinza     -    e cria um array com numpy
        imagem_PIL = Image.open(caminho_imagem).convert('L')
        imagem_numpy = np.array(imagem_PIL,'uint8')

        #pega o nome de cada imagem slitando pelo "." no nome dela
        face_id  = int(os.path.split(caminho_imagem)[-1].split(".")[1])
        # aplica o classificador nas imagens armazenadas em um array do numpy afinal estamos em um loop
        faces = classificador.detectMultiScale(imagem_numpy)

        #para cada imagem/face ele vai dar um append adicionando nas listas
        for (x,y,w,h) in faces:
            dataset_faces.append(imagem_numpy[y:y+h,x:x+w])
            ids.append(face_id )
    #retorna os valores das imagens e das id's para serem gravadas no arquivo de treino
    return dataset_faces,ids




print ("\n [INFO] Treinando as faces, aguarde...")

#chama nossa função
faces,ids = imagensIds(caminho_dataset)
#chama o trainer com as arrays gravadas no numpy
reconhecedor.train(faces, np.array(ids))

# Salva o treino feito em trainer/trainer.yml
reconhecedor.write('trainer/trainer.yml')

# Printa o numero de faces treinadas e termina o programa!
print('\n [INFO] {0} faces treinadas, processo concluido.'.format(len(np.unique(ids))))
