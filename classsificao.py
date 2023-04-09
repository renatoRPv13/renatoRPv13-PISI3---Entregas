import pandas as pd
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from copy import deepcopy
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from mlxtend.plotting import plot_decision_regions
from functools import reduce
from sklearn import tree
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
import scikitplot as skplt
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from sklearn.metrics import silhouette_score
import seaborn as sns
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import decomposition
from scipy.spatial.distance import cdist, pdist
import warnings


warnings.filterwarnings("ignore")
st.sidebar.title("Pontuações de precisão")
st.title("Painel de recomendação")
st.subheader("""Site da prefeitura: Trabalho proposto""")
st.subheader("Nosso trabalho será avaliar quais atributos influenciam nas informações de execução orçamentária e "
             "financeira detalham as receitas e despesas da prefeitura de São Paulo, permitindo ao cidadão "
             "acompanhar os gastos com diárias e passagens, os empenhos realizados, entre outras informações "
             "orçamentárias e financeiras do órgão construir um modelo preditivo para realizar previsões estatisticas")
st.subheader('Usaremos os conjuntos de dados abaixo')
avalicao = pd.read_csv(r'data/baseDadosExecucao2019.csv')
avalicaoDesenpenho = pd.read_csv(r'data/baseDadosExecucao2020.csv')
infomacao = pd.read_csv(r'data/baseDadosExecucao2021.csv')
registratroOrcanento = pd.read_csv(r'data/baseDadosExecucao2022.csv')
orcamento = pd.read_csv(r'data/basedadosexecucao2019.csv')
valorOrcado = pd.read_csv(r'data/basedadosexecucao2020.csv')
disponivel = pd.read_csv(r'data/basedadosexecucao2021.csv')
categoria = pd.read_csv(r'data/categorizadosCorrelacao.csv')
categoria_2 = pd.read_csv(r'data/categorizadosCorrelacao_2.csv')
categoria_3 = pd.read_csv(r'data/categorizadosCorrelacao_3.csv')

set1 = list(avalicao.columns.values)
set2 = list(avalicaoDesenpenho.columns.values)
set3 = list(infomacao.columns.values)
set4 = list(registratroOrcanento.columns.values)
set5 = list(orcamento.columns.values)
set6 = list(valorOrcado.columns.values)
set7 = list(disponivel.columns.values)
set8 = list(categoria.columns.values)
all_columns = [set1, set2, set3, set4, set5, set6, set7, set8]


count_columns = [avalicao.size, avalicaoDesenpenho.size, infomacao.size, registratroOrcanento.size, orcamento.size,
                 valorOrcado.size, disponivel.size, categoria.size]
columns_header = ['Avaliação', 'Avaliação Desenpenho', 'Informação', 'Registratro Orcanento', 'Orçamento',
                  'Valo Orcado', 'Disponivel', 'Categoria']
# removendo valores nulos
# Removendo as linhas com valores missing
avalicao.dropna(inplace=True)
avalicaoDesenpenho.dropna(inplace=True)
infomacao.dropna(inplace=True)
registratroOrcanento.dropna(inplace=True)
orcamento.dropna(inplace=True)
valorOrcado.dropna(inplace=True)
disponivel.dropna(inplace=True)
categoria.dropna(inplace=True)
categoria_2.dropna(inplace=True)
categoria_3.dropna(inplace=True)

valorOrcado.duplicated()
disponivel.duplicated()
valorOrcado.drop_duplicates()
disponivel.drop_duplicates()
infomacao.drop_duplicates()
avalicao.drop_duplicates()
avalicaoDesenpenho.drop_duplicates()
registratroOrcanento.drop_duplicates()
categoria.drop_duplicates()
categoria_2.drop_duplicates()
categoria_3.drop_duplicates()

# Verificando valores missing
avalicao.isna().sum()
avalicaoDesenpenho.isna().sum()
infomacao.isna().sum()
registratroOrcanento.isna().sum()
orcamento.isna().sum()
valorOrcado.isna().sum()
disponivel.isna().sum()
categoria.isna().sum()
categoria_2.isna().sum()
categoria_3.isna().sum()


count_columns_1 = [avalicao.size, avalicaoDesenpenho.size, infomacao.size, registratroOrcanento.size, orcamento.size,
                 valorOrcado.size, disponivel.size, categoria.size]
d = {'Tabela Nomes': columns_header, 'Linhas e Colunas': count_columns_1,'Informações do conjunto de dados': all_columns}
df = pd.set_option('max_colwidth', 200)
df = pd.DataFrame(d)
df

# analisando os dados base em recursos, desenpenho,bairro,ensino municipal,educação,região, imd
# resultado com base no
sigla_Orgao = infomacao.groupby(['Sigla_Orgao'], as_index=False)
cont_Sigla = sigla_Orgao['Cd_Orgao'].count()
resultado = infomacao.groupby(['Sigla_Orgao', 'papa'], as_index=False)
cont_resultado = resultado['Cd_Orgao'].count()

merge = pd.merge(cont_Sigla, cont_resultado, on='Sigla_Orgao', how='inner') # how='left')
merge['i'] = round(merge['Cd_Orgao_y'] / merge['Cd_Orgao_x'], 2)

merge = merge[['Sigla_Orgao', 'papa', 'i']]

sigla_1 = merge.loc[merge['Sigla_Orgao'] == 'FMD']

sigla_2 = merge.loc[merge['Sigla_Orgao'] == 'SGM']
sigla_3 = merge.loc[merge['Sigla_Orgao']=='SMSUB']
sigla_4 = merge.loc[merge['Sigla_Orgao']=='SG']
fig = plt.figure()
ax = fig.add_subplot(111)

sigla_1.set_index('papa', drop=True, inplace=True)
sigla_2.set_index('papa', drop=True, inplace=True)
sigla_3.set_index('papa', drop=True, inplace=True)
sigla_4.set_index('papa', drop=True, inplace=True)
sigla_1.plot(kind='bar', ax=ax, width=0.3, position=0)
sigla_2.plot(kind='bar', color='#2ca02c', ax=ax, width=0.3, position=1)
sigla_3.plot(kind='bar', ax=ax, width=0.3, position=0)
sigla_4.plot(kind='bar', color='#2ca02c', ax=ax, width=0.3, position=1)
plt.xlabel('Resultados')
plt.ylabel('resultado_status')

st.subheader('Análise Exploratória')
st.subheader('Shape')
st.text(df.shape)
plt.title('Relação entre o desempenho das Siglas de cada Orgão codigo do orgão')
plt.legend(['FMD', 'SGM','SMSUB','SG'])
plt.show()
st.subheader('Resultado da base união dos dataframe usando o merge')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
st.markdown('A Sigla do Orgao representam as secretarias.(sigla_1 sigla_2)')
# st.set_option('deprecation.showPyplotGlobalUse', False)

# #resultados com base no atributo Cd_Despesa
Cd_Despesa = valorOrcado.groupby(['Ds_Fonte'], as_index=False)
cont_Cd_Despesa = Cd_Despesa['Cd_Orgao'].count()
resultado_Cd_Despesa = valorOrcado.groupby(['Ds_Fonte', 'papa'], as_index=False)
resultado_Despesa = resultado_Cd_Despesa['Cd_Orgao'].count()

merge = pd.merge(cont_Cd_Despesa, resultado_Despesa, on='Ds_Fonte', how='left')
merge['_'] = round(merge['Cd_Orgao_y'] / merge['Cd_Orgao_x'], 2)
merge = merge[['Ds_Fonte', 'papa', '_']]

merge.set_index(['Ds_Fonte', 'papa']).unstack().plot(kind='barh', stacked=True)
box = ax.get_position()
ax.get_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.ylabel('Despesas')
plt.xlabel('Resultdos')
plt.legend(['Projetos', 'Despesas Correntes', 'INVESTIMENTOS', 'Aplicações Diretas'], loc='center left',
           bbox_to_anchor=(1, 0.80))
plt.show()
st.subheader('Resultados das despesas')
st.pyplot()

st.sidebar.title("Atualmente os diversos métodos para construir modelos iniciais estão sendo explorados e com base em suas acurácias"
                 " e nas necessidades do módulo, o melhor modelo será implementado")
st.subheader('Despesa e disponivel para empenhado liquido: e dados concatenados')
receitas = disponivel[['Cd_Despesa', 'Disponivel', 'Vl_Liquidado', 'Vl_Pago']]
# concatená-los.
df = pd.concat([receitas], ignore_index=True)
st.write(df)

valorOrcado.duplicated()
disponivel.duplicated()
valorOrcado.drop_duplicates()
disponivel.drop_duplicates()

# Exibir informações estatísticas sobre o conjunto de dados.
st.subheader('Informações estatísticas sobre os dados')
st.write(df.describe())
st.subheader('A correlação entre todas as variáveis')
st.write(df.corr())

st.subheader('Comando utilizado para verificar as linhas finais do DataFrame')
st.write(infomacao.tail())

st.subheader('Avaliar o período dos dados coletados')
inicio = pd.to_datetime(infomacao['Cd_Exercicio']).dt.date.min()
fim = pd.to_datetime(infomacao['Cd_Exercicio']).dt.date.max()
st.write('Período dos dados - De:', inicio, 'Até:',fim)

infomacao = pd.get_dummies(infomacao)
print(infomacao.head())

st.subheader("Categorização e mapeamento dos dados de strings para inteiros")
dataset = pd.read_csv("data/dataframe.csv", index_col=0, encoding="latin-1")
st.write(set(dataset["Administracao"]))

u = deepcopy(dataset)

le = preprocessing.LabelEncoder()

le.fit(dataset["Administracao"])
dataset["Administracao"] = le.transform(dataset["Administracao"])
le_name_mapping_1 = dict(zip(le.classes_, le.transform(le.classes_)))

le.fit(dataset["Sigla_Orgao"])
dataset["Sigla_Orgao"] = le.transform(dataset["Sigla_Orgao"])
le_name_mapping_2 = dict(zip(le.classes_, le.transform(le.classes_)))

le.fit(dataset["Ds_Orgao"])
dataset["Ds_Orgao"] = le.transform(dataset["Ds_Orgao"])
le_name_mapping_3 = dict(zip(le.classes_, le.transform(le.classes_)))


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


import json

a = json.dumps(le_name_mapping_1, default=np_encoder, indent=True, ensure_ascii=False)
b = json.dumps(le_name_mapping_2, default=np_encoder, indent=True, ensure_ascii=False)
c = json.dumps(le_name_mapping_3, default=np_encoder, indent=True, ensure_ascii=False)
ff = '{'
ff += f'\n"Administracao": [\n{a}\n],\n"Sigla_Orgao": [\n{b}\n],\n"Ds_Orgao": [\n{c}\n]'
ff += '\n}'
f = open("mapa_dataframa.json", "w")
f.write(ff)
f.close()
dataset.to_csv("dados_dados_categorizados.csv")
print(dataset)

st.subheader('Dessa forma podemos agrupar os valores e identificar se há algum valor discrepante.')
st.write(dataset.groupby(['Administracao','Sigla_Orgao','Ds_Orgao','Cd_Despesa','Vl_Orcado_Ano','Vl_Reduzido','Disponivel','Vl_Congelado','Vl_EmpenhadoLiquido','Vl_Liquidado','Vl_Pago']).size())
# Em seguida convertemos o campo em float
dataset['Cd_Despesa'] = dataset['Cd_Despesa'].astype(np.float64)
dataset['Vl_Orcado_Ano'] = dataset['Vl_Orcado_Ano'].astype(np.float64)
dataset["Vl_Reduzido"] = [float(str(i).replace(",", "")) for i in dataset["Vl_Reduzido"]]
dataset['Vl_Reduzido'] = dataset['Vl_Reduzido'].astype(np.float64)
dataset["Disponivel"] = [float(str(i).replace(",", "")) for i in dataset["Disponivel"]]
dataset['Disponivel'] = dataset['Disponivel'].astype(np.float64)
dataset["Vl_Congelado"] = [float(str(i).replace(",", "")) for i in dataset["Vl_Congelado"]]
dataset['Vl_Congelado'] = dataset['Vl_Congelado'].astype(np.float64)
dataset["Vl_EmpenhadoLiquido"] = [float(str(i).replace(",", "")) for i in dataset["Vl_EmpenhadoLiquido"]]
dataset['Vl_EmpenhadoLiquido'] = dataset['Vl_EmpenhadoLiquido'].astype(np.float64)
dataset["Vl_Liquidado"] = [float(str(i).replace(",", "")) for i in dataset["Vl_Liquidado"]]
dataset['Vl_Liquidado'] = dataset['Vl_Liquidado'].astype(np.float64)
dataset["Vl_Pago"] = [float(str(i).replace(",", "")) for i in dataset["Vl_Pago"]]
dataset['Vl_Pago'] = dataset['Vl_Pago'].astype(np.float64)
print(dataset.isnull().sum())
print(dataset.info())

st.subheader('Agora iremos avaliar os outliers das colunas que são númericas OUTLIERS são valores discrepantes que estão bem'
             ' acima ou bem abaixo dos outros valores Vamos carregar em uma lista as variaveis que são do tipo INT64 E FLOAT64')
variaveis_numericas = []
for i in dataset.columns[0:48].tolist():
    if dataset.dtypes[i] == 'int64' or dataset.dtypes[i] == 'float64':
        st.write(i, ':' , dataset.dtypes[i])
        variaveis_numericas.append(i)
st.write(variaveis_numericas)

# Preparando os dados para o plot
# Cria uma cópia do dataset original
df = dataset.copy()

# Listas vazias para os resultados
continuous = []
categorical = []

# Loop pelas colunas
for c in df.columns[:-1]:
    if df.nunique()[c] >= 30:
        continuous.append(c)
    else:
        categorical.append(c)

st.text(continuous)

st.subheader('Gráficos para variáveis numéricas.')
# Plot das variáveis contínuas

# Tamanho da área de plotagem
fig = plt.figure(figsize = (12,8))

# ‘Loop’ pelas variáveis contínuas
for i, col in enumerate(continuous):
    plt.subplot(3, 3, i + 1);
    df.boxplot(col);
    plt.tight_layout()
st.pyplot()

st.subheader(' Transformação de log nas variáveis contínuas.')
df[continuous] = np.log1p(1 + df[continuous])

# Plot das variáveis contínuas

# Tamanho da área de plotagem
fig = plt.figure(figsize = (12,8))

# ‘Loop’ pelas variáveis contínuas
for i,col in enumerate(continuous):
    plt.subplot(3,3,i+1);
    df.boxplot(col);
    plt.tight_layout()
st.pyplot()
le = preprocessing.LabelEncoder()
for i in dataset.columns:
    if dataset[i].dtype == object:
        dataset[i] = le.fit_transform(dataset[i])
    else:
        pass

dataset_encoded = le.fit_transform(dataset.columns)

from pylab import rcParams

st.subheader('Mapa de correlação entre os dados')


def plot_correlation(data):
    rcParams['figure.figsize'] = 10, 10
    fig = plt.figure()
    sns.heatmap(data.corr(), annot=True, fmt=".2f")
    plt.show()
    plt.title("MATRIZ DE CORRELAÇÃO", fontsize=14)
    fig.savefig('corr.png')


df = dataset.drop(columns=[
    "Sigla_Orgao",
    "Administracao",
    "Cd_Despesa",
    "Ds_Orgao"
])
st.pyplot(plot_correlation(df))

sns.lmplot(x = "Vl_Orcado_Ano", y = "Cd_Despesa", data = dataset);
st.pyplot()

st.subheader('Separando os dados treinamento e teste')
X = dataset.drop(columns=[
    "Administracao",
    "Sigla_Orgao",
    "Ds_Orgao",
    "Cd_Despesa",
    "Vl_Orcado_Ano"
])
y = dataset["Administracao"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


st.subheader('Criando o objeto KNN')
classifier = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
st.text(y_pred)

st.subheader('Validando a acurácia e precisão da predição dos dados')

st.text("Confusion matrix:\n")
st.text(confusion_matrix(y_test, y_pred))
st.text("\nClassification report:\n")
st.text(classification_report(y_test, y_pred))
st.markdown("Accuracy: ", accuracy_score(y_test, y_pred))

st.subheader('Medido o "score" da predição')
knn = KNeighborsClassifier(n_neighbors=8, metric="euclidean")
k = knn.fit(X_train, y_train)
pred_i = knn.predict(X_test)
k_score = k.score(X_test, y_test)
st.text(k_score)

st.subheader('Procurando o melhor valor para K através da taxa de erro')
error = []

for i in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=i, metric="euclidean")
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 30), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=5)
plt.title('Taxa de erro do valor de K')
plt.xlabel('Valor de K')
plt.ylabel('Erro médio')
st.pyplot()
def knn_comparison(X, y, k):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X, y)
    value = 1.5
    width = 0.75
    plt.figure(figsize=(10, 8))
    plot_decision_regions(
        X.to_numpy(),
        y,
        clf=clf,
        legend=2
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Knn com K=" + str(k))
    plt.show()
X = dataset.drop(columns=[
    "Administracao",
    "Sigla_Orgao",
    "Ds_Orgao",
    "Cd_Despesa",
    "Vl_Orcado_Ano"
])
y = dataset["Administracao"].values
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)
le.fit(dataset["Administracao"])
dataset["Administracao"] = le.transform(dataset["Administracao"])
X2_train, X2_test, y2_train, y2_test = train_test_split(X, y, test_size=0.4)

st.text(" ")
st.subheader('Modelo de previsão de classificação com algotimos de arvores')
dfs = [categoria, categoria_2, categoria_3]
df_final = reduce(lambda left, right: pd.merge(left, right, on='id_orgao'), dfs)
df_final['funcao'].value_counts()
df_final = df_final.drop(['despesa'], axis=1)

# convertendo tabela final em dados categóricos
le = preprocessing.LabelEncoder()
df_final = df_final.apply(le.fit_transform)

# Decision Tree
X = df_final.loc[:, df_final.columns != 'funcao']
y = df_final['funcao']

xTrain, xTest, yTrain, yTest = train_test_split(X, y, train_size=0.75)
# criterio gini
dt = tree.DecisionTreeClassifier(criterion='gini')
dt = dt.fit(xTrain, yTrain)
train_pred = dt.predict(xTrain)
test_pred = dt.predict(xTest)
st.header("Accuracy da Arvore de Decisão: {0:.3f}".format(metrics.accuracy_score(yTest, test_pred)),"\n")

# Regressão de aumento de gradiente
X = df_final.loc[:, df_final.columns != 'funcao']
y = df_final['funcao']
xTrain, xTest, yTrain, yTest = train_test_split(X, y, train_size=0.75)
# Construtor do modelo
# valores de grid para o hiperparamentos n_estimators=100...
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=0, loss='huber')
gb = gb.fit(xTrain, yTrain)
st.header("Accuracy Gradient Boosting Regressor :{0:.3f}".format(gb.score(xTest, yTest)))

# Random Forest
X = df_final.loc[:, df_final.columns != 'funcao']
y = df_final['funcao']
xTrain, xTest, yTrain, yTest = train_test_split(X, y, train_size=0.75)

# Construtor do modelo
# valores de grid para o hiperparamentos n_estimators=10
rf = RandomForestClassifier(n_estimators=10, random_state=33)
rf = rf.fit(xTrain, yTrain)
train_pred = rf.predict(xTrain)
test_pred = rf.predict(xTest)
st.header("Accuracy classificação Random Forest :{0:.3f}".format(metrics.accuracy_score(yTest, test_pred)), "\n")
st.markdown(
    ' Observação: usamos árvores de decisão para prever o desempenho da aplicação dos recursos e a precisão não era boa, então usamos florestas aleatórias e técnicas de aumento de gradiente para melhorar a precisão.')
st.markdown('Aqui estão os mapas de calor para as pontuações de precisão das técnicas que usamos')
# Função para traçar a precisão

def plot_accuracy(model):
    train_sizes, train_scores, \
        test_scores = learning_curve(model, xTrain, yTrain, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), verbose=0)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.figure()
    plt.gca().invert_yaxis()
    plt.grid()
    plt.ylim(0.0, 1.1)
    st.write(plt.title("Accuracy Plot = Plotagem de Precisão"))
    st.write(plt.xlabel("Testing = teste"))
    st.write(plt.ylabel("Accuracy = Precisão%"))
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
    st.write(plt.plot(train_sizes, test_mean, 'bo-', color="r", label="Test Score"))
    # Decision Tree
    st.subheader('Accuracy for Decision Tree = Precisão para Árvore de Decisão')
    plot_accuracy(dt)
    st.pyplot()
    # Decision Tree
if st.sidebar.button(' Confusion Matrix for decision tree = Matriz de confusão para árvore de decisão'):
    st.header(' Confusion Matrix for decision tree = Matriz de confusão para árvore de decisão')
    y_pred = cross_val_predict(dt, xTest, yTest)
# normalizando
    skplt.metrics.plot_confusion_matrix(yTest, y_pred, normalize=True)
    plt.show()
st.pyplot()
# Gradient Boosting Aumento de Gradiente
if st.sidebar.button('Confusion Matrix for gradient boosting = Matriz de confusão para aumento de gradiente'):
    st.header(' Confusion Matrix for gradient boosting = Matriz de confusão para aumento de gradiente')
    y_pred = cross_val_predict(gb, xTest, yTest)
    y_pred = np.absolute(y_pred)
    skplt.metrics.plot_confusion_matrix(yTest, y_pred.round(), normalize=True)
    plt.show()
st.pyplot()
# Random Forest
if st.sidebar.button(' Confusion Matrix for random forest = Matriz de confusão para floresta aleatória '):
    st.header(' Confusion Matrix for random forest = Matriz de confusão para floresta aleatória')
    y_pred = cross_val_predict(rf, xTest, yTest)
    skplt.metrics.plot_confusion_matrix(yTest, y_pred, normalize=True)
   # plt.show()
st.pyplot()

sns.lmplot(x = "id_orgao", y = "funcao", data = df_final);
st.pyplot()

###################################################################################
st.subheader('Machine Clustering')
st.subheader('Selecionando apenas as classs que possuem recursos')
df = dataset.loc[
    (dataset['Administracao'] > 0.0) & (dataset['Sigla_Orgao'] > 0.0) & (dataset['Ds_Orgao'] > 0.0) & (
            dataset['Cd_Despesa'] > 0.0) & (
            dataset['Vl_Orcado_Ano'] > 0.0)]
st.write(dataset.describe())

st.subheader('Visualizando os outliers da classe Administracao')
sns.set(rc={'figure.figsize': (5, 4)})
outliers = pd.read_csv(r"data/dado_categorizados.csv")
sns.boxplot(outliers['Ds_Orgao'])
st.pyplot()

st.subheader('Visualizando os outliers dos dados do orçamento.')
sns.set(rc={'figure.figsize': (15, 7)})
orca = pd.DataFrame(data=np.random.random(size=(5, 5)),
                    columns=['Administracao', 'Sigla_Orgao', 'Ds_Orgao', 'Cd_Despesa', 'Vl_Orcado_Ano'])
sns.boxplot(x="variable", y="value", data=pd.melt(orca), palette="Blues_r").figure.savefig('pos_outliers')
st.pyplot()

st.subheader('Usando matriz de correlação')
matriz_corr = dataset.corr()
st.write(matriz_corr)


df_1 = pd.read_csv(r'data/dado_categorizados.csv')

# trocando virgula por ponto nos campos de valores
df_1['Cd_Despesa'] = df_1['Cd_Despesa'].apply(lambda x: str(x).replace(',', '.'))
df_1['Vl_Orcado_Ano'] = df_1['Vl_Orcado_Ano'].apply(lambda x: str(x).replace(',', '.'))

# convertendo as colunas para tipo numérico
df_1['Cd_Despesa'] = df_1['Cd_Despesa'].astype('float64')
df_1['Vl_Orcado_Ano'] = df_1['Vl_Orcado_Ano'].astype('float64')
df_1['Cd_Despesa'].fillna((df_1['Cd_Despesa'].median()), inplace=True)

X_train = df_1.drop("Vl_Orcado_Ano", axis=1)
Y_train = df_1["Vl_Orcado_Ano"]
X_test = df_1.drop("Ds_Orgao", axis=1).copy()

# Número de valores exclusivos em todas as colunas são as esperadas
# Vamos primeiro dividir os recursos por tipo para entender melhor os dados
for dados in ['Cd_Orgao', 'Administracao', 'Sigla_Orgao', 'Ds_Orgao', 'Cd_Despesa', 'Vl_Orcado_Ano']:
    print("\n\n", df_1[dados].value_counts())
num_cursos_novo = df_1['Administracao']
Q1 = num_cursos_novo.quantile(.25)
Q3 = num_cursos_novo.quantile(.75)
IIQ = Q3 - Q1
limite_inferior = Q1 - 1.5 * IIQ
limite_superior = Q3 + 1.5 * IIQ

selecao = (num_cursos_novo >= limite_inferior) & (num_cursos_novo <= limite_superior)
novo_df = df_1[selecao]

# Remove os registros com valores NA e remove as duas primeiras colunas (não são necessárias)
novo_df = novo_df.iloc[0:, 2:9].dropna()

# Checando se há valores missing
novo_df.isnull().values.any()

st.subheader('Obtém os valores dos atributos. Neste caso as variaveis foram carregadas ''como categorias '
             '(object) entao iremos extrair os valores.')
valor = novo_df.values
st.write(valor)

st.subheader('Coleta uma amostra de 1% dos dados para não comprometer a memória do computador')
amostra_1, amostra_2 = train_test_split(valor, test_size=0.1)
st.text(amostra_1.shape)

st.subheader('Foi Aplicado redução de dimensionalidade Transforma as 6 variáveis em 2 variaveis principais. '
             'Esse método utiliza Algebra Linear pra identificar semelhança entre os dados e assim "juntar" '
             'as variaveis, medindo a semelhança pela variância.')


pca = decomposition.PCA(n_components=2)
pca.fit(amostra_1)
X = pca.transform(amostra_1)
st.write(X)

st.subheader('Criando um objeto de k-means e fornecendo o número de clusters')
agrupador = KMeans(n_clusters=2)
agrupador.fit(X)
labels = agrupador.labels_
st.text(labels)

st.subheader('Determinando um range de K')
k_range = range(1, 12)
st.subheader('Aplicando o modelo K-Means para cada valor de K (esta célula pode levar bastante tempo para ser executada')
k_means_var = [KMeans(n_clusters = k).fit(X) for k in k_range]
st.subheader('Ajustando o centróide do cluster para cada modelo')
centroids = [X.cluster_centers_ for X in  k_means_var]
st.text(centroids)
st.subheader('Calculando a distância euclidiana de cada ponto de dado para o centróide')
k_euclid = [cdist(X, cent, 'euclidean') for cent in centroids]
dist = [np.min(ke, axis=1) for ke in k_euclid]
st.write(dist)

st.subheader('Soma dos quadrados das distâncias dentro do cluster')
soma_quadrados_intra_cluster = [sum(d**2) for d in dist]
st.text(soma_quadrados_intra_cluster)

st.subheader('Soma total dos quadrados')
soma_total = sum(pdist(X)**2)/X.shape[0]
st.text(soma_total)

st.subheader('Soma dos quadrados entre clusters')
soma_quadrados_inter_cluster = soma_total - soma_quadrados_intra_cluster
st.text(soma_quadrados_inter_cluster)

st.subheader('Curva de Elbow')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(k_range, soma_quadrados_inter_cluster/soma_total * 100, 'b*-')
ax.set_ylim((0, 100))
plt.grid(True)
plt.xlabel('Número de Clusters')
plt.ylabel('Percentual de Variância Explicada')
plt.title('Variância Explicada x Valor de K')
st.pyplot()

# Criando um modelo com K = 8
modelo_v1 = KMeans(n_clusters=8)
st.text(modelo_v1.fit(X))

st.subheader('Visualizando os clusters')
fig = go.Figure()
fig.add_trace(go.Scatter(x=novo_df['Cd_Despesa'], y=novo_df['Vl_Orcado_Ano'],
                         mode='markers',
                         marker=dict(color=agrupador.labels_.astype(np.float)),
                         text=labels))

st.pyplot()

st.subheader('Visualizando o número total de clusters')
st.text(np.unique(agrupador.labels_))

st.subheader('Quando reduzimos muito o epsilon, todos os dados são consideradas ruídos. Para se organizar'
             ' em clusters é necessário um número mínimo de 15 vizinhos.')
agrupador = DBSCAN(eps=3, min_samples=15, metric='euclidean')
st.text(agrupador)

st.subheader("Observações: eps é a máxima distância entre os pontos/ mínimo de pontos é igual a 10/ "
             "a métrica de distância considerada'manhattan'")
agrupador.fit(novo_df)
st.text(agrupador.labels_)


st.subheader('Construindo um modelo com 3 clusters')
kmode = KModes(n_clusters=3, init="random", n_init=5, verbose=1)
clusters = kmode.fit_predict(novo_df)
st.text(clusters)

st.subheader('Dataframe inserindo clusters.')
novo_df.insert(0, "Cluster", clusters, True)
st.write(novo_df)

st.subheader('Dataframe chamando pelo seu índice do clusters zero.')
cluster_0 = novo_df.loc[novo_df['Cluster'] == 0]
st.dataframe(cluster_0.head())

st.subheader('Dataframe chamando pelo seu índice do clusters um.')
cluster_1 = novo_df.loc[novo_df['Cluster'] == 1]
st.text(cluster_1.head())
st.text(cluster_1.shape)

st.subheader('Dataframe chamando pelo seu índice do clusters dois.')
cluster_2 = novo_df.loc[novo_df['Cluster'] == 2]
st.text(cluster_2.head())
st.text(cluster_2.shape)

st.subheader(' Usando o K-prototypes')

# K-Prototypes
kproto = KPrototypes(n_clusters=4, init='Cao', verbose=2)
clusters = kproto.fit_predict(novo_df, categorical=[1, 2, 3])
st.text(clusters)

st.subheader('Centróides do modelo treinado')
st.text(kproto.cluster_centroids_)

st.subheader('Estatísticas de treinamento')
st.text(kproto.cost_)
st.text(kproto.n_iter_)

st.subheader('Contagem de cada cluster')
st.text(pd.Series(clusters).value_counts())

st.subheader('Coeficiente de Silhueta')
st.subheader('Verificando melhor número de clusters')
faixa_n_clusters = [i for i in range(2, 10)]
valores_silhueta = []
for k in faixa_n_clusters:
    agrupador = KMeans(n_clusters=k, random_state=10)  # random state para inicializar sempre no mesmo local
    labels = agrupador.fit_predict(
        df_1[['Cd_Orgao', 'Administracao', 'Sigla_Orgao', 'Ds_Orgao', 'Cd_Despesa', 'Vl_Orcado_Ano']])
    media_silhueta = silhouette_score(
        df_1[['Cd_Orgao', 'Administracao', 'Sigla_Orgao', 'Ds_Orgao', 'Cd_Despesa', 'Vl_Orcado_Ano']], labels)
    valores_silhueta.append(media_silhueta)
st.write(valores_silhueta)

plt.figure(figsize=(7, 4))
sns.lineplot(x=faixa_n_clusters, y=valores_silhueta, palette='crest')
sns.set_style('darkgrid')
plt.title("COEFICIENTE DE SILHUETA", fontsize=12)
plt.xlabel('Valores de ‘k’', fontsize=13)
plt.ylabel('Coeficiente de Silhueta', fontsize=13)
plt.savefig('coeficiente_silhueta.png', transparent=True)
st.pyplot()

st.subheader('Inicializando e Computando o KMeans com o valor de 3 clusters')
X1 = novo_df[['Ds_Orgao', 'Cd_Despesa']].iloc[:, :].values
algorithm = (KMeans(n_clusters=3))
algorithm.fit(X1)
st.subheader('Visualizando os grupos criados e seus centroides:')
labels2 = algorithm.labels_  # Visualizando os grupos criados e seus centroides:
centroids2 = algorithm.cluster_centers_
st.write(centroids2)

h = 0.02
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])
plt.figure(1, figsize=(7, 7))
plt.clf()
Z2 = Z.reshape(xx.shape)
plt.scatter(x='Ds_Orgao', y='Cd_Despesa', data=novo_df, c=labels2, s=15)
plt.scatter(x=centroids2[:, 0], y=centroids2[:, 1], s=50, c='yellow', alpha=1)
plt.title("CLUSTER 1", fontsize=14)
plt.ylabel('Conceito Médio do eixo X do orgão'), plt.xlabel('Recursos do eixo Y da despessa')
plt.savefig('k-means1.png', transparent=True)
st.pyplot()
st.text(" ")
st.subheader('Inicializando e Computando o KMeans com o valor de 3 clusters')
X1 = novo_df[['Ds_Orgao', 'Vl_Orcado_Ano']].iloc[:, :].values
algorithm = (KMeans(n_clusters=3))
algorithm.fit(X1)
st.subheader('Visualizando os grupos criados e seus centroides:')
labels2 = algorithm.labels_  # Visualizando os grupos criados e seus centroides:
centroids2 = algorithm.cluster_centers_
st.write(centroids2)


h = 0.02
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])
plt.figure(1, figsize=(7, 7))
plt.clf()

Z2 = Z.reshape(xx.shape)
plt.scatter(x='Ds_Orgao', y='Vl_Orcado_Ano', data=novo_df, c=labels2, s=15)
plt.scatter(x=centroids2[:, 0], y=centroids2[:, 1], s=50, c='yellow', alpha=1)
plt.title("CLUSTER 2", fontsize=14)
plt.ylabel('Conceito Médio do eixo X do orgão'), plt.xlabel('Recursos do eixo Y valor do Orçamento do Ano')
plt.savefig('k-means1.png', transparent=True)
st.pyplot()
st.text(" ")
st.subheader('Inicializando e Computando o KMeans com o valor de 3 clusters')
X1 = novo_df[['Ds_Orgao', 'Vl_Orcado_Ano']].iloc[:, :].values
algorithm = (KMeans(n_clusters=3))
algorithm.fit(X1)
st.subheader('Visualizando os grupos criados e seus centroides:')
labels2 = algorithm.labels_  # Visualizando os grupos criados e seus centroides":
centroids2 = algorithm.cluster_centers_
st.write(centroids2)

h = 0.02
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])
plt.figure(1, figsize=(7, 7))
plt.clf()

Z2 = Z.reshape(xx.shape)
plt.scatter(x='Ds_Orgao', y='Vl_Orcado_Ano', data=novo_df, c=labels2, s=15)
plt.scatter(x=centroids2[:, 0], y=centroids2[:, 1], s=50, c='yellow', alpha=1)
plt.title("CLUSTER 3", fontsize=14)
plt.ylabel('Conceito Médio do eixo X do orgão'), plt.xlabel('Recursos do eixo Y clusters da Administracao')
plt.savefig('k-means1.png', transparent=True)
st.pyplot()

# Baseia-se no agrupamento de "clusters" de baixo para cima, combinando a cada etapa dois "clusters" que contenham o par
# mais próximo de elementos que ainda não pertencem ao mesmo "cluster".
st.subheader('Agrupamento de ligação única:Em estatística, o agrupamento de '
             'ligação única é um dos vários métodos de agrupamento hierárquico.')

novo_df1 = pd.read_csv(r"data/dados_orcamento.csv", header='infer')

names = novo_df1['Administracao']
Y = novo_df1['Ds_SubFuncao']
X = novo_df1.drop(['Administracao', 'Ds_SubFuncao'], axis=1)
Z = hierarchy.linkage(X.values)
dn = hierarchy.dendrogram(Z, labels=names.tolist(), orientation='right')
st.pyplot()

st.subheader('Link Completo (MAX)')
Z = hierarchy.linkage(X.values, 'complete')
dn = hierarchy.dendrogram(Z, labels=names.tolist(), orientation='right')
st.pyplot()

st.subheader('Média do Grupo')
Z = hierarchy.linkage(X.values, 'average')
dn = hierarchy.dendrogram(Z, labels=names.tolist(), orientation='right')
st.pyplot()

data_1 = pd.read_csv(r'data/dados_categorizados.csv', sep=";")
freme = pd.read_csv(r'data/dataframe.csv', sep=";")

st.markdown('.Resultados')
st.subheader('Inserindo os clusters no dataframe.')
df_reindexed = novo_df.reset_index()
df_reindexed['cluster'] = algorithm.labels_
st.write(df_reindexed.head())
st.text(" ")
st.subheader('Descrevendo informações do cluster zero')
st.write(cluster_0.describe())

st.subheader("Despesas soma das despesas com investimentos")
orca_cluster0 = cluster_0['Cd_Despesa'].sum()
orca_cluster1 = cluster_1['Cd_Despesa'].sum()
orca_cluster2 = cluster_2['Cd_Despesa'].sum()
st.text(orca_cluster0)
st.text(orca_cluster1)
st.text(orca_cluster2)

st.subheader("A media do valor orçado")
orca_medio0 = cluster_0['Vl_Orcado_Ano'].mean()
orca_medio1 = cluster_1['Vl_Orcado_Ano'].mean()
orca_medio2 = cluster_2['Vl_Orcado_Ano'].mean()
st.text(orca_medio0)
st.text(orca_medio1)
st.text(orca_medio2)


st.subheader("Valor medio da classe Ds orgao")
orca_ds_med0 = cluster_0['Ds_Orgao'].sum()
orca_ds_med1 = cluster_1['Ds_Orgao'].sum()
orca_ds_med2 = cluster_2['Ds_Orgao'].sum()
st.text(orca_ds_med0)
st.text(orca_ds_med1)
st.text(orca_ds_med2)

st.subheader("Valor medio da classe das  Siglas do orgao")
orca_sigla_med0 = cluster_0['Sigla_Orgao'].mean()
orca_sigla_med1 = cluster_1['Sigla_Orgao'].mean()
orca_sigla_med2 = cluster_2['Sigla_Orgao'].mean()
st.text(orca_sigla_med0)
st.text(orca_sigla_med1)
st.text(orca_sigla_med2)

st.subheader('Toda soma dos numeros do Cluster em grupos ')
incluster_0 = [0, 593, orca_cluster0, orca_medio0,  orca_ds_med0]
incluster_1 = [1, 866, orca_cluster1, orca_medio1,  orca_ds_med1]
incluster_2 = [2, 88, orca_cluster2, orca_medio2,  orca_ds_med2]

clusters = [incluster_0, incluster_1, incluster_2]
grupos = pd.DataFrame(clusters,
                      columns=['Cluster', 'Vl_Orcado_Ano', 'Cd_Despesa', 'Ds_Orgao',
                               'Sigla_Orgao'])
st.write(grupos.head())

grupos['Despesa (%)'] = grupos['Cd_Despesa'] / grupos['Cd_Despesa'].sum() * 100
grupos['Vl_Orcado_Ano (%)'] = grupos['Vl_Orcado_Ano'] / grupos['Vl_Orcado_Ano'].sum() * 100

st.subheader('Grupo de cluster 1')
st.write(grupos.head())
st.line_chart(grupos)
st.subheader('Transposição de matriz.')
st.write(grupos.transpose())
st.line_chart(grupos)

st.subheader('Grupo de cluster 2')
st.write(grupos.head())
st.line_chart(grupos)



