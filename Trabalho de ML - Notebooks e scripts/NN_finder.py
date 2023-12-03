# %%
import tensorflow as tf
import pandas as pd
import numpy as np

# Importando os dados
df_lag_cp = pd.read_csv('data/bem_comportadas_laguardia.csv')
df_lag_li = pd.read_csv('data/livres_laguardia.csv')
df_van_cp = pd.read_csv('data/bem_comportadas_vanessa.csv')
df_van_li = pd.read_csv('data/livres_vanessa.csv')
df_bru_cp = pd.read_csv('data/bem_comportadas_bruno.csv')
df_bru_li = pd.read_csv('data/livres_bruno.csv')

# %%
# Separando os dados em X e Y
X_lag_cp = df_lag_cp.drop('pose', axis=1)
Y_lag_cp = df_lag_cp['pose'].astype('category')
X_lag_li = df_lag_li.drop('pose', axis=1)
Y_lag_li = df_lag_li['pose'].astype('category')
X_van_cp = df_van_cp.drop('pose', axis=1)
Y_van_cp = df_van_cp['pose'].astype('category')
X_van_li = df_van_li.drop('pose', axis=1)
Y_van_li = df_van_li['pose'].astype('category')
X_bru_cp = df_bru_cp.drop('pose', axis=1)
Y_bru_cp = df_bru_cp['pose'].astype('category')
X_bru_li = df_bru_li.drop('pose', axis=1)
Y_bru_li = df_bru_li['pose'].astype('category')

# %%
# Juntando os dados de treino e teste
X_train = pd.concat([X_lag_cp, X_van_cp, X_bru_cp])
Y_train = pd.concat([Y_lag_cp, Y_van_cp, Y_bru_cp])
X_test = pd.concat([X_lag_li, X_van_li, X_bru_li])
Y_test = pd.concat([Y_lag_li, Y_van_li, Y_bru_li])

# %%
# ===== Constantes ===== #
NUM_NEURONS = 7                 # Número de neurônios nas camadas escondidas (2**NUM_NEURONS)
STEP_NUM_NEURONS = 1            # Passo de variação do número de neurônios
NUM_HIDDEN_LAYERS = 0           # Número de camadas escondidas
STEP_HIDDEN_LAYERS = 1          # Passo de variação do número de camadas escondidas
NUM_EPOCHS = 100                # Número de épocas
STEP_NUM_EPOCHS = 100           # Passo de variação do número de épocas
NUM_EPOCHS_PER_LOG = 20         # Número de épocas por log
BATCH_SIZE = 5                  # Tamanho do batch (2**BATCH_SIZE)
STEP_BATCH_SIZE = 1             # Passo de variação do tamanho do batch
NUM_SEEDS = 10                  # Número de seeds para testar em cada configuração
BEST_LOSS_THRESHOLD = 0.05      # Threshold para salvar o modelo
MODEL_SAVING = True             # Salvar o modelo
np.random.seed(42)

# %%
# ===== Modelo ===== #
def instantiate_model(seed, num_neurons, num_hidden_layers):
    tf.random.set_seed(seed)
    
    model_layers = []
    model_layers.append(tf.keras.layers.Dense(2**num_neurons, activation='relu', input_shape=(X_lag_cp.shape[1],)))
    for i in range(num_hidden_layers):
        model_layers.append(tf.keras.layers.Dense(2**num_neurons, activation='relu'))
    model_layers.append(tf.keras.layers.Dense(13, activation='softmax'))

    model = tf.keras.models.Sequential(model_layers)

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    return model

# %%
# ===== Treinamento ===== #
def train_models(num_neurons, num_hidden_layers, num_epochs, batch_size):
    best_loss_overall = np.inf
    best_accuracy_overall = 0.1
    for seed in range(NUM_SEEDS):
        model = instantiate_model(seed, num_neurons, num_hidden_layers)

        total_epochs = 0
        best_loss = np.inf
        best_accuracy = 0.1
        best_model = None
        for i in range(num_epochs // NUM_EPOCHS_PER_LOG):
            model.fit(X_train, Y_train, epochs=NUM_EPOCHS_PER_LOG, batch_size=2**batch_size, verbose=0)
            total_epochs += NUM_EPOCHS_PER_LOG
            loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
            print('Epochs: {}'.format(total_epochs))
            print('Loss: {}'.format(loss))
            print('Accuracy: {}'.format(accuracy))

            # Salvando o melhor modelo
            if loss / accuracy < best_loss / best_accuracy:
                best_loss = loss
                best_accuracy = accuracy
                best_model = model
            else:
                print(f'Early stopping at {total_epochs} epochs')
                break
        
        # Salvando o melhor modelo
        if (best_loss / best_accuracy < best_loss_overall / best_accuracy_overall) * (1 + BEST_LOSS_THRESHOLD):
            best_loss_overall = best_loss
            best_accuracy_overall = best_accuracy
            if MODEL_SAVING:
                best_model.save(f'models/best_NN_Loss{best_loss:.3f}_Acc{best_accuracy}_Seed{seed}')
                print('Model saved')
    
    return best_loss_overall, best_accuracy_overall

# %%
# ===== Grid Search ===== #
def grid_search(num_neurons, num_hidden_layers, num_epochs, batch_size):
    params = [num_neurons, num_hidden_layers, num_epochs, batch_size]
    steps = [STEP_NUM_NEURONS, STEP_HIDDEN_LAYERS, STEP_NUM_EPOCHS, STEP_BATCH_SIZE]

    best_loss = np.inf
    best_accuracy = 0.1
    best_params = []
    for i in range(len(params)):                                    # Para cada parâmetro
        for j in range(-1, 2, 2):                                   # Para cada direção
            params[i] += steps[i] * j                               # Altera o parâmetro
            loss, accuracy = train_models(*params)
            if loss / accuracy < best_loss / best_accuracy:         # Se a perda for menor
                best_loss = loss
                best_accuracy = accuracy
                best_params = params.copy()
            params[i] -= steps[i] * j                               # Volta o parâmetro
    
    return best_loss, best_accuracy, best_params
    
def step_up(old_params, new_params):
    params_updated = old_params.copy()
    for i in range(len(old_params)):
        diff = new_params[i] - old_params[i]
        params_updated[i] += diff * 2

    return params_updated
        
# %%
# ===== Execução ===== #
best_params = [NUM_NEURONS, NUM_HIDDEN_LAYERS, NUM_EPOCHS, BATCH_SIZE]
if __name__ == "__main__":
    best_loss, best_accuracy = train_models(*best_params)           # Treina o modelo inicial
    gs_loss, gs_accuracy, gs_params = grid_search(*best_params)     # Faz o grid search

    while gs_loss / gs_accuracy < best_loss / best_accuracy:        # Enquanto o grid search estiver melhorando
        best_loss = gs_loss
        best_accuracy = gs_accuracy
        best_params = gs_params.copy()

        gs_params = step_up(best_params, gs_params)                 # Avança os parâmetros do grid search
        gs_loss, gs_accuracy, gs_params = grid_search(*gs_params)
    
    print('===== Model training finished =====')
    print(f'Best loss: {best_loss}')
    print(f'Best accuracy: {best_accuracy}')
    print(f'Best params: {best_params}')
    with open('best_params.txt', 'w') as f:
        f.write(f'Best loss: {best_loss}\n')
        f.write(f'Best accuracy: {best_accuracy}\n')
        f.write(f'Best params: {best_params}\n')
    print('===== End =====')