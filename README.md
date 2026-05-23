# Tech Challenge Fase 4 - Sistema de Triagem de Obesidade

Projeto preditivo e analítico desenvolvido para a fase 4 da pós-graduação em Data Analytics da FIAP.

## Diferencial Técnico (Remoção de Data Leakage)
Durante a fase de desenvolvimento, identificamos que a inclusão das variáveis de Peso e Altura causava um vazamento de dados crítico, visto que a variável alvo (Obesity) é derivada diretamente do cálculo matemático do IMC. O modelo original atingia 96% de acurácia apenas replicando essa lógica óbvia.

Para entregar uma solução com real valor de negócio para o ambiente hospitalar, removemos o Peso e a Altura do treinamento. O modelo final foi treinado puramente com base em histórico genético e hábitos comportamentais do paciente. O algoritmo alcançou 87% de acurácia real, superando com folga a meta de 75% estipulada no regulamento.

## Estrutura do Projeto
* train.py: Script automatizado e vetorizado que realiza o pré-processamento dos dados (via ColumnTransformer) e treina o classificador.
* app.py: Aplicação web desenvolvida em Streamlit, dividida em duas abas principais (Sistema de Triagem Preditiva e Painel de Insights de Negócio).
* Dockerfile: Manifesto de containerização da aplicação para implantação em produção.
* Obesity.csv: Base de dados utilizada para o estudo epidemiológico.

## Como Executar

1. Instale as dependências contidas no arquivo de requisitos:
pip install -r requirements.txt

2. Execute o treinamento para gerar os artefatos do modelo:
python train.py

3. Inicie a aplicação local do Streamlit:
streamlit run app.py
