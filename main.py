import  streamlit as st
import  yfinance as yf
from datetime import  date
import  pandas as pd
#from fbprophet import  Prophet
#from  fbprobhet.plot import  plot_pltly plot_components_plotly
from  plotly import  graph_objects as go
import webbrowser



#url = 'https://raw.githubusercontent.com/renatoRPv13/renatoRPv13-PISI3---Entregas/f1b73c9189e947c897213928ce705baa996463a2/basedadosexecucao2019.csv'

# Abra o URL em uma nova guia, se uma janela do navegador já estiver aberta.
#webbrowser.open_new_tab(url)
#  Abra o URL em uma nova janela, aumentando a janela, se possível
#webbrowser.open_new(url)

df1 = pd.read_csv('https://raw.githubusercontent.com/renatoRPv13/renatoRPv13-PISI3---Entregas/f1b73c9189e947c897213928ce705baa996463a2/basedadosexecucao2019.csv', sep=',', encoding='utf8')
st.write(df1)

df2 = pd.read_csv('https://raw.githubusercontent.com/renatoRPv13/renatoRPv13-PISI3---Entregas/main/basedadosexecucao2020.csv', sep=',', encoding='utf8')
st.write(df2)

df3 = pd.read_csv('https://raw.githubusercontent.com/renatoRPv13/renatoRPv13-PISI3---Entregas/main/basedadosexecucao2021.csv', sep=',', encoding='utf8')
st.write(df3)

df4 = pd.read_csv('https://raw.githubusercontent.com/renatoRPv13/renatoRPv13-PISI3---Entregas/main/basedadosexecucao20221122.csv', sep=',', encoding='utf8')
st.write(df4)





