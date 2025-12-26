import streamlit as st
import pandas as pd
import numpy as np
import scikit_posthocs as sp
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
import matplotlib.pyplot as plt

def carregar_dados(train_path, test_path):
    """
        Carrega e prepara o dataset Spectf_train e Spectf_test.
     """
    df_train = pd.read_csv(train_path, header=None)
    df_test = pd.read_csv(test_path, header=None)
    df = pd.concat([df_train, df_test], ignore_index=True)
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    return StandardScaler().fit_transform(X), y

# Carregando os dados
train_path = "~/PycharmProjects/Francisco/spectf+heart/SPECTF.train"
test_path = "~/PycharmProjects/Francisco/spectf+heart/SPECTF.test"

# Processando os dados
X, y = carregar_dados(train_path, test_path)
#df_processado = pd.DataFrame(X)  # Convertendo array numpy para DataFrame
#st.markdown(df_processado.head(10).to_html())
#st.markdown(df_processado.head())
#st.markdown("Shape X:", X.shape)
#st.markdown("Shape y:", y.shape)
#st.markdown("X:",X.shape)
#st.sidebar("y:", y.shape)
st.write(X)
st.write(y)

# Calcula a distância kernelizada entre um ponto x e um centróide v
# Usando kernel Gaussiano ponderado por pesos λ e desvios sigmas
def kernel_distance(x, v, lambdas, sigmas):
    return sum(
        lambdas[j] * (1 - np.exp(-((x[j] - v[j])**2) / (2 * sigmas[j]**2)))
        for j in range(len(x))
    )
# Retorna matriz D onde D[i, k] = distância do ponto i para centróide k
def calcular_matriz_distancias(X, centroids, lambdas, sigmas):
    return np.array([
        [kernel_distance(x, centroids[k], lambdas, sigmas) for k in range(len(centroids))]
        for x in X
    ])

# Atribui cada ponto ao cluster com menor distância
def atribuir_clusters(D):
    return np.argmin(D, axis=1)

# Recalcula cada centróide como a média dos pontos do cluster
def atualizar_centroides(X, U, K):
    return np.array([
        X[U == k].mean(axis=0) if np.any(U == k) else np.zeros(X.shape[1])
        for k in range(K)
    ])

# Atualiza os pesos λ com base na variabilidade intra-cluster
def atualizar_lambdas(X, U, centroids, sigmas):
    p = X.shape[1]
    num_all = np.zeros(p)
    for j in range(p):
        for k in range(len(centroids)):
            cluster_k = X[U == k]
            if len(cluster_k) > 0:
                num_all[j] += np.sum(1 - np.exp(-((cluster_k[:, j] - centroids[k][j])**2) / (2 * sigmas[j]**2)))
    prod = np.prod(num_all)**(1/p)
    return np.array([prod / (val + 1e-10) for val in num_all])

# Soma da dissimilaridade kernelizada de todos os pontos para seus centróides
def calcular_J(X, U, centroids, lambdas, sigmas):
    return sum(kernel_distance(X[i], centroids[U[i]], lambdas, sigmas) for i in range(X.shape[0]))

def executar_vkcmk(X, y, K_range, n_runs=50, max_iter=10):
    p = X.shape[1]
    sigmas = np.std(X, axis=0) + 1e-6
    resultados = {}

    for K in K_range:
        melhor_sil = -1
        melhor_resultado = {}

        print(f"\n🔍 Testando K = {K}")

        for run in range(n_runs):
            try:
                idx = np.random.choice(X.shape[0], K, replace=False)
                centroids = X[idx]
                lambdas = np.ones(p)
                U = np.zeros(X.shape[0], dtype=int)
                J_hist = []  # ⬅️ Histórico da função objetivo

                for _ in range(max_iter):
                    D = calcular_matriz_distancias(X, centroids, lambdas, sigmas)
                    U = atribuir_clusters(D)
                    centroids = atualizar_centroides(X, U, K)
                    lambdas = atualizar_lambdas(X, U, centroids, sigmas)
                    J_atual = calcular_J(X, U, centroids, lambdas, sigmas)
                    J_hist.append(J_atual)

                if len(set(U)) == 1:
                    st.write(f"⚠️ Run {run}: apenas 1 cluster gerado.")
                    continue

                sil = silhouette_score(X, U)

                if sil > melhor_sil:
                    melhor_sil = sil
                    melhor_resultado = {
                        'K': K,
                        'U': U.copy(),
                        'centroids': centroids.copy(),
                        'lambdas': lambdas.copy(),
                        'J': J_hist[-1],
                        'J_hist': J_hist.copy(),  # ⬅️ salvar histórico
                        'silhouette': sil
                    }

                st.write(f"✅ Run {run} | Silhueta = {sil:.4f} | J final = {J_hist[-1]:.2f}")

            except Exception as e:
                st.write(f"❌ Run {run} falhou: {e}")

        if melhor_resultado:
            resultados[K] = melhor_resultado
            st.write(f"✅ K = {K} finalizado | Melhor Silhueta: {melhor_sil:.4f}")
        else:
            st.write(f"⚠️ Nenhum resultado viável para K = {K}")

    return resultados

# # Carregar seus dados
# from sklearn.datasets import load_digits
#
# # Carregar o conjunto de dados
# X, y = load_digits(return_X_y=True)

# Agora você pode executar a função
resultados = executar_vkcmk(X, y, K_range=[2, 3, 4, 5], n_runs=50, max_iter=10)

st.write(f"### Resultados para {resultados.keys()}")
#df_modelo = pd.DataFrame(resultados)
#Mostrar as métricas principais para cada K
for k in resultados.keys():
    st.write(f"### Resultados para K = {k}")

    # Criar um DataFrame com as métricas principais
    metricas = {
        'Silhueta': resultados[k]['silhouette'],
        'J final': resultados[k]['J'],
        'Número de clusters': k
    }

    # Converter para DataFrame
    df_metricas = pd.DataFrame([metricas])

    # Exibir as métricas
    st.dataframe(df_metricas.round(4))

    # Opcionalmente, mostrar os centroids
    st.write("Centroids:")
    st.dataframe(pd.DataFrame(resultados[k]['centroids']).round(4))

#Plotando Silhueta para cada K e identificando o melhor
Ks = list(resultados.keys())
st.write(f"🔍 Resultados para K = {Ks}:")
sils = [resultados[k]["silhouette"] for k in Ks]

plt.figure(figsize=(6,4))
plt.plot(Ks, sils, marker='o', linestyle='-')
plt.title("Silhueta × Número de Clusters (K)")
plt.xlabel("Número de Clusters (K)")
plt.ylabel("Índice de Silhueta")
plt.grid(True)
st.pyplot(plt)

# Melhor K (maior silhueta)
melhor_K = Ks[np.argmax(sils)]
melhor_resultado = resultados[melhor_K]
st.write(f"🔎 Melhor número de clusters (K*) = {melhor_K}")

from sklearn.metrics import adjusted_rand_score

ari = adjusted_rand_score(y, melhor_resultado["U"])
st.write(f"✅ Índice de Rand Corrigido (ARI) para K* = {melhor_K}: {ari:.4f}")

st.write(f"📍 Protótipos (v_k) para K = {melhor_K}:")
st.write(melhor_resultado["centroids"])


st.write("📊 Vetor de pesos de relevância λ:")
st.write(melhor_resultado["lambdas"])

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Obter matriz de confusão
cm = confusion_matrix(y, melhor_resultado["U"])

# Visualizar matriz
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="viridis")
plt.title(f"Matriz de Confusão - VKCM-K (K = {melhor_resultado['K']})")
plt.xlabel("Cluster atribuído (VKCM-K)")
plt.ylabel("Classe real (a priori)")
st.pyplot(plt)

# Plot da função J por iteração para o melhor resultado
J_hist = melhor_resultado["J_hist"]

plt.figure(figsize=(6, 4))
plt.plot(range(1, len(J_hist) + 1), J_hist, marker='o')
plt.title("Função Objetivo J × Iterações")
plt.xlabel("Iteração")
plt.ylabel("Função Objetivo J")
plt.grid(True)
st.pyplot(plt)

"""
    Carrega dados da segunda questão.
"""

# Imports essenciais
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
import seaborn as sns

from scipy.stats import friedmanchisquare, t, sem

from sklearn import preprocessing
# Primeiro, importe o StandardScaler (você já tem isso no código)
from sklearn.preprocessing import StandardScaler

# Caminhos dos arquivos no Drive
train_path = "~/PycharmProjects/Francisco/spectf+heart/SPECTF.train"
test_path = "~/PycharmProjects/Francisco/spectf+heart/SPECTF.test"
# 📁 Leitura e preparação dos dados
df_train = pd.read_csv(train_path, header=None) # Use train_path instead of just the filename
df_test = pd.read_csv(test_path, header=None)  # Use test_path instead of just the filename
df = pd.concat([df_train, df_test], ignore_index=True)
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values
st.write(f"Número de classes: {np.unique(y)}")

# Normalização\scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# print("Dados normalizados.")
# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
st.markdown("Dados normalizados.")
# # Crie uma instância do StandardScaler
# scaler = StandardScaler()
#
# # Agora você pode usar o scaler
# X_scaled = scaler.fit_transform(X)

# Função de validação cruzada 30x10

def executar_cv(classificador, X, y, nome="Modelo"):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accs, precisions, recalls, f1s = [], [], [], []

    for _ in range(30):
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            classificador.fit(X_train, y_train)
            y_pred = classificador.predict(X_test)

            accs.append(accuracy_score(y_test, y_pred))
            precisions.append(precision_score(y_test, y_pred))
            recalls.append(recall_score(y_test, y_pred))
            f1s.append(f1_score(y_test, y_pred))

    st.markdown(f"\n✅ {nome} executado com sucesso.")
    return {
        "accuracy": accs,
        "precision": precisions,
        "recall": recalls,
        "f1": f1s,
        "error": [1 - a for a in accs]
    }

def executar_cv_1(classificador, X, y, nome="Modelo"):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accs, precisions, recalls, f1s = [], [], [], []

    for _ in range(30):
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            classificador.fit(X_train, y_train)
            y_pred = classificador.predict(X_test)

            accs.append(accuracy_score(y_test, y_pred))
            precisions.append(precision_score(y_test, y_pred))
            recalls.append(recall_score(y_test, y_pred))
            f1s.append(f1_score(y_test, y_pred))

    return {
        "accuracy": accs,
        "precision": precisions,
        "recall": recalls,
        "f1": f1s,
        "error": [1 - a for a in accs]
    }

#  Execução dos modelos
resultados_dict = {}
resultados_dict["Gaussiano"] = executar_cv_1(GaussianNB(), X_scaled, y, "Gaussiano")
resultados_dict["KNN"] = executar_cv_1(KNeighborsClassifier(n_neighbors=5), X_scaled, y, "KNN")
resultados_dict["RegLog"] = executar_cv_1(LogisticRegression(C=1.0, penalty='l2', solver='liblinear'), X_scaled, y, "Regressao Logistica")

voto = VotingClassifier(estimators=[
    ('gnb', GaussianNB()),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('lr', LogisticRegression(solver='liblinear'))
], voting='hard')

# Criar um DataFrame com os resultados médios
# resultados_medios = {}
# for modelo in resultados_dict:
#     metricas = {}
#     for metrica in ['accuracy', 'precision', 'recall', 'f1', 'error']:
#         metricas[metrica] = np.mean(resultados_dict[modelo][metrica])
#     resultados_medios[modelo] = metricas
#
# # Converter para DataFrame
# df_resultados = pd.DataFrame(resultados_medios).T
#
# # Exibir no Streamlit
# st.dataframe(df_resultados.round(4))

# resultados_dict["VotoMajor"] = executar_cv_1(voto, X_scaled, y, "Voto Majoritario")
# st.dataframe(resultados_dict)
for modelo in resultados_dict:
    st.write(f"### Resultados para {modelo}")
    df_modelo = pd.DataFrame(resultados_dict[modelo])
    st.dataframe(df_modelo.round(4))

#  Intervalo de Confiança

def resumo_ic(medidas):
    media = np.mean(medidas)
    intervalo = t.interval(0.95, len(medidas)-1, loc=media, scale=sem(medidas))
    return media, intervalo

for modelo in resultados_dict:
    print(f"\n {modelo}")
    for metrica in ['f1', 'precision', 'recall', 'error']:
        media, ic = resumo_ic(resultados_dict[modelo][metrica])
        #st.dataframe(f"{metrica.upper()}: {media:.3f} ± {ic[1] - media:.3f}")
        st.write(f"{metrica.upper()}: {media:.3f} ± {ic[1] - media:.3f}")


#import matplotlib.pyplot as plt

# 🧪 Teste de Friedman + Nemenyi
def teste_friedman_nemenyi(resultados_dict, metrica):
    nomes = list(resultados_dict.keys())
    dados = [resultados_dict[n][metrica] for n in nomes]
    df = pd.DataFrame(dados).T
    df.columns = nomes

    print(f"\n📊 Teste de Friedman para a métrica: {metrica.upper()}")
    stat, p = friedmanchisquare(*[df[c] for c in df.columns])
    print(f"→ Estatística: {stat:.4f} | p-valor: {p:.4f}")

    if p < 0.05:
        print("✅ Diferença significativa detectada! Aplicando Nemenyi...\n")
        nemenyi = sp.posthoc_nemenyi_friedman(df.values)
        nemenyi.index = nemenyi.columns = nomes
        return nemenyi
    else:
        print("⚠️ Nenhuma diferença significativa entre classificadores.")
        return None


resultado_nemenyi_f1 = teste_friedman_nemenyi(resultados_dict, 'f1')
st.write(resultado_nemenyi_f1)

if resultado_nemenyi_f1 is not None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(resultado_nemenyi_f1, annot=True, fmt=".3f", cmap="Blues")
    plt.title("Teste de Nemenyi — F1-score")
    st.pyplot(plt)


#Curva de aprendizado do Gaussiano

def avaliar_gaussiano_learning_curve(X, y, test_size=0.2, step=0.05, seed=42):
    global y_train_full, X_test, X_train_full, y_test
    resultados = {"treino_pct": [], "erro": [], "precision": [], "recall": [], "f1": []}
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)

    for train_idx, test_idx in sss.split(X, y):
        X_train_full, X_test = X[train_idx], X[test_idx]
        y_train_full, y_test = y[train_idx], y[test_idx]

    n_train = len(X_train_full)

    for pct in np.arange(0.05, 1.0, step):
        n_amostras = int(pct * n_train)
        X_treino = X_train_full[:n_amostras]
        y_treino = y_train_full[:n_amostras]

        modelo = GaussianNB()
        modelo.fit(X_treino, y_treino)
        y_pred = modelo.predict(X_test)

        resultados["treino_pct"].append(pct*100)
        resultados["erro"].append(1 - accuracy_score(y_test, y_pred))
        resultados["precision"].append(precision_score(y_test, y_pred))
        resultados["recall"].append(recall_score(y_test, y_pred))
        resultados["f1"].append(f1_score(y_test, y_pred))

    return pd.DataFrame(resultados)

# Executar curva de aprendizado
df_curve = avaliar_gaussiano_learning_curve(X_scaled, y)

# Plotar curva de aprendizado
plt.figure(figsize=(12, 6))
plt.plot(df_curve["treino_pct"], df_curve["erro"], label="Erro")
plt.plot(df_curve["treino_pct"], df_curve["precision"], label="Precisão")
plt.plot(df_curve["treino_pct"], df_curve["recall"], label="Recall")
plt.plot(df_curve["treino_pct"], df_curve["f1"], label="F1-score")

plt.title("Curva de Aprendizado — Bayesiano Gaussiano")
plt.xlabel("Porcentagem do Conjunto de Treinamento (%)")
plt.ylabel("Métrica no Conjunto de Teste")
plt.legend()
plt.grid(True)
st.pyplot(plt)