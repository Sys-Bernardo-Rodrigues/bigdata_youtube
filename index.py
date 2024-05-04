import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from IPython.display import display

df = pd.read_csv("dataset.csv", encoding='latin1')


#Limpeza e Pré-processamento dos Dados
#Neste passo, você deve limpar os dados e prepará-los para análise. Isso pode incluir 
#tratamento de valores nulos, conversão de tipos de dados, etc.
# Tratar valores nulos
dataset = df.dropna()


# Renomear a coluna corretamente
data = df.copy()
data.rename(columns=lambda x: x.replace('video views', 'video_views'), inplace=True)


# Converter tipos de dados, se necessário
dataset['visualizacoes'] = dataset['visualizacoes'].astype(int)
dataset['inscritos'] = dataset['inscritos'].astype(int)

#Análise Exploratória de Dados (EDA)
#Realize uma análise exploratória dos dados para entender melhor as tendências e padrões.
# Análise exploratória básica
print(dataset.describe())

# Visualizações vs Inscritos
import matplotlib.pyplot as plt
plt.scatter(dataset['visualizacoes'], dataset['inscritos'])
plt.xlabel('Visualizações')
plt.ylabel('Inscritos')
plt.title('Visualizações vs Inscritos')
plt.show()



