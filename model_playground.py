import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import cv2

# ===== Data ===== #
DFS = {
    'lag_cp' : pd.read_csv('data/bem_comportadas_laguardia.csv'),
    'lag_li' : pd.read_csv('data/livres_laguardia.csv'),
    'van_cp' : pd.read_csv('data/bem_comportadas_vanessa.csv'),
    'van_li' : pd.read_csv('data/livres_vanessa.csv'),
    'bru_cp' : pd.read_csv('data/bem_comportadas_bruno.csv'),
    'bru_li' : pd.read_csv('data/livres_bruno.csv')
}
DFS_X = {
    'lag_cp' : DFS['lag_cp'].drop('pose', axis=1),
    'lag_li' : DFS['lag_li'].drop('pose', axis=1),
    'van_cp' : DFS['van_cp'].drop('pose', axis=1),
    'van_li' : DFS['van_li'].drop('pose', axis=1),
    'bru_cp' : DFS['bru_cp'].drop('pose', axis=1),
    'bru_li' : DFS['bru_li'].drop('pose', axis=1)
}
DFS_Y = {
    'lag_cp' : DFS['lag_cp']['pose'].astype('string'),
    'lag_li' : DFS['lag_li']['pose'].astype('string'),
    'van_cp' : DFS['van_cp']['pose'].astype('string'),
    'van_li' : DFS['van_li']['pose'].astype('string'),
    'bru_cp' : DFS['bru_cp']['pose'].astype('string'),
    'bru_li' : DFS['bru_li']['pose'].astype('string')
}

# ===== Constants ===== #
USERS = ['Laguardia', 'Vanessa', 'Bruno']
SENSORS = ['p' + str(n).zfill(2) for n in range(12)]
POSES = [str(pose) for pose in range(13)]
MODELS = ['KNN', 'MLPClassifier', 'RandomForestClassifier', 'DecisionTreeClassifier']

# ===== Helper Functions ===== #
def get_Xy(users_selected_, poses_selected_, sensors_selected_=None):
    users_convert = {
        'Laguardia' : 'lag',
        'Vanessa' : 'van',
        'Bruno' : 'bru'
    }
    users = [users_convert[user] for user in users_selected_]

    X_train = pd.concat([DFS_X[user + '_cp'] for user in users])
    y_train = pd.concat([DFS_Y[user + '_cp'] for user in users])
    X_test = pd.concat([DFS_X[user + '_li'] for user in users])
    y_test = pd.concat([DFS_Y[user + '_li'] for user in users])

    for pose in POSES:
        if pose not in poses_selected_:
            X_train = X_train[y_train != pose]
            y_train = y_train[y_train != pose]
            X_test = X_test[y_test != pose]
            y_test = y_test[y_test != pose]

    if sensors_selected_ is not None:
        for col in X_train.columns:
            if col not in sensors_selected_:
                X_train = X_train.drop(col, axis=1)
                X_test = X_test.drop(col, axis=1)

    return X_train, y_train, X_test, y_test

# ===== App ===== #
st.title('Buscador de Modelos')

users_selected = st.multiselect('Selecione os usuários:', USERS, default=USERS)

poses_selected = st.multiselect('Selecione as poses:', POSES, default=POSES)

with st.expander('Ver poses'):
    img = cv2.imread('poses.jpg')
    st.image(img, channels='BGR')

sensors_selected = st.multiselect('Selecione os sensores:', SENSORS, default=SENSORS)

model = st.selectbox('Selecione o modelo:', MODELS)

X_train, y_train, X_test, y_test = [], [], [], []
if st.button('Buscar'):
    with st.spinner('Acessando dados...'):
        X_train, y_train, X_test, y_test = get_Xy(users_selected, poses_selected)
        st.plotly_chart(px.histogram(y_train, color=y_train, title='Distribuição das poses no conjunto de treino'))
        st.plotly_chart(px.histogram(y_test, color=y_test, title='Distribuição das poses no conjunto de teste'))


    with st.spinner('Buscando...'):
        st.subheader('Modelo selecionado: ' + model)
        match model:
            case 'KNN':
                from sklearn.neighbors import KNeighborsClassifier
                model = KNeighborsClassifier(n_neighbors=3)

            case 'MLPClassifier':
                from sklearn.neural_network import MLPClassifier
                model = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)

            case 'RandomForestClassifier':
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100)
            
            case 'DecisionTreeClassifier':
                from sklearn.tree import DecisionTreeClassifier
                model = DecisionTreeClassifier()
        
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.write('Acurácia:', model.score(X_test, y_test))

        st.write('Matriz de Confusão:')
        matrix = pd.crosstab(y_test, y_pred, rownames=['Real'], colnames=['Predito'])
        st.plotly_chart(px.imshow(matrix, color_continuous_scale='Blues'))