import pandas as pd
from pandas_profiling import ProfileReport
from pandas_profiling.utils.cache import cache_file
from statsmodels.tsa.vector_ar import output
from IPython.core.display import display
from  IPython.core.display import  display_html
from ipywidgets import HTML, Button, widgets
"""
#Data:
#https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
#http://dados.recife.pe.gov.br/dataset/acidentes2021-de-transito-com-e-sem-vitimas
#
print('Digite o nome do arquivo:')
data_file = input()
try:
    df = pd.read_csv(f'data/{data_file}.csv', sep=',')
except:
    df = pd.read_csv(f'data/{data_file}.csv', sep=';')

profile = ProfileReport(df, title=f"{data_file} Dataset")
profile.to_file(f"{data_file}.html")
"""
data_file = input()
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
#profile = ProfileReport(df,title=f"{data_file}explorative=True")
#pf = ProfileReport(df)
#df.to_widgets()
#profile.to_widgets()
#df.to_file(output_file='output.html')
#profile = ProfileReport(df, title=f"{data_file} Dataset")
profile = ProfileReport(df,title="Demo Exploratory Analysis",explorative=True)
profile.to_file(output_file='output.html')
display(profile)
#from ipywidgets import HTML, Button, widgets
#ModuleNotFoundError: No module named 'ipywidgets'

