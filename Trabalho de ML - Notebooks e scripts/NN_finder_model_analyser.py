import pandas as pd
import numpy as np
import tensorflow as tf
import os

# ===== Constantes ===== #
MODELS_PATH = 'models/'
MAX_MODELS = 10

class model_file:
    def __init__(self, name, loss, accuracy, idx):
        self.name = name
        self.loss = loss
        self.accuracy = accuracy
        self.idx = idx
    
    def __str__(self):
        return f'Modelo {self.idx} > loss: {self.loss:.3f} - accuracy: {self.accuracy:.3f}'
    
    def load_model(self):
        return tf.keras.models.load_model(MODELS_PATH + self.name)


models_list = []
for i, file_name in enumerate(os.listdir(MODELS_PATH)):
    
    props = file_name.split('_')
    loss = float(props[2].replace('Loss', ''))
    accuracy = float(props[3].replace('Acc', ''))
    
    this_model = model_file(file_name, loss, accuracy, i)
    models_list.append(this_model)

# ===== Interface ===== #

last_last_inp = None
last_inp = None
inp = None
backing = False
if __name__ == '__main__':
    print('1 - Listar modelos por loss/accuracy')
    print('2 - Listar modelos por loss')
    print('3 - Listar modelos por accuracy')
    print('4 X - Carregar modelo X')
    print('5 - Voltar')

    while True:
        last_last_inp = last_inp
        last_inp = inp
        if not backing:
            print('\nEscolha uma opção:')
            inp = input('>>> ')

        backing = False
        match inp:
            case '1':
                print('Listando modelos por loss/accuracy: ')
                models_list.sort(key=lambda x: x.loss/x.accuracy)
                i = 0
                for model in models_list:
                    print(model)
                    i += 1
                    if i == MAX_MODELS:
                        break
            case '2':
                print('Listando modelos por loss: ')
                models_list.sort(key=lambda x: x.loss)
                i = 0
                for model in models_list:
                    print(model)
                    i += 1
                    if i == MAX_MODELS:
                        break
            case '3':
                print('Listando modelos por accuracy: ')
                models_list.sort(key=lambda x: x.accuracy, reverse=True)
                i = 0
                for model in models_list:
                    print(model)
                    i += 1
                    if i == MAX_MODELS:
                        break
            case '5':
                if last_inp[0] == '4':
                    inp = last_last_inp
                    backing = True
                else:
                    break
            case _:
                if inp[0] == '4':
                    try:
                        model_fil = None
                        for model in models_list:
                            if model.idx == int(inp[2:]):
                                model_fil = model
                                break
                        print(f'Carregando modelo {model_fil.idx}...')
                        model = model_fil.load_model()
                        print(model.summary())
                    except:
                        model = model_fil.load_model()
                        print(model.summary())


