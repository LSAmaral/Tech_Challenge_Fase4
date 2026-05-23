import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Triagem de Obesidade - Hospital", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/obesity_pipeline.pkl")
    mapping = joblib.load("models/label_mapping.pkl")
    return model, mapping

try:
    pipeline, label_mapping = load_artifacts()
except FileNotFoundError:
    st.error("Execute o script 'train.py' primeiro para gerar os novos modelos!")
    st.stop()

st.title("Portal Hospitalar - Inteligência e Triagem Precoce de Obesidade")
st.markdown("Ferramenta preditiva baseada em hábitos e genética para suporte ao diagnóstico precoce antes do ganho de peso crítico.")

# Dicionários de Suporte
map_gender = {"Feminino": "Female", "Masculino": "Male"}
map_yes_no = {"Sim": "yes", "Não": "no"}
map_caec_calc = {"Não": "no", "Às vezes": "Sometimes", "Frequentemente": "Frequently", "Sempre": "Always"}
map_mtrans = {
    "Transporte Público": "Public_Transportation", "Caminhando": "Walking",
    "Automóvel": "Automobile", "Motocicleta": "Motorbike", "Bicicleta": "Bike"
}
map_output = {
    'Insufficient_Weight': 'Peso Insuficiente (Abaixo do Peso)', 'Normal_Weight': 'Peso Normal',
    'Overweight_Level_I': 'Sobrepeso Grau I', 'Overweight_Level_II': 'Sobrepeso Grau II',
    'Obesity_Type_I': 'Obesidade Grau I', 'Obesity_Type_II': 'Obesidade Grau II',
    'Obesity_Type_III': 'Obesidade Grau III (Mórbida)'
}

tab_pred, tab_dash = st.tabs(["Triagem Preventiva por Hábitos", "Painel de Insights (Visão de Negócio)"])

with tab_pred:
    st.header("Formulário de Anamnese Comportamental e Genética")
    st.caption("Insira os dados de rotina do paciente. Peso e altura foram removidos para evitar viés matemático e focar na prevenção de risco.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender_ui = st.selectbox("Gênero do Paciente", list(map_gender.keys()))
        age = st.slider("Idade (Anos)", 14, 61, 25)
        family_history_ui = st.selectbox("Histórico Familiar de Excesso de Peso?", list(map_yes_no.keys()), 
                                         help="Indica se o paciente possui parentes de primeiro grau com histórico de obesidade.")
        mtrans_ui = st.selectbox("Meio de transporte principal utilizado", list(map_mtrans.keys()), help="Principal modalidade de deslocamento urbano do paciente.")
        
    with col2:
        favc_ui = st.selectbox("Consome alimentos altamente calóricos com frequência?", list(map_yes_no.keys()), help="Ingestão frequente de fast-food ou alimentos hipercalóricos.")
        caec_ui = st.selectbox("Consome algum alimento entre as refeições?", list(map_caec_calc.keys()), help="Hábito de petiscar fora do cronograma das refeições principais.")
        scc_ui = st.selectbox("O paciente monitora as calorias diárias?", list(map_yes_no.keys()), help="Prática de controle calórico ativo.")
        smoke_ui = st.selectbox("O paciente é fumante?", list(map_yes_no.keys()))
        
    with col3:
        fcvc = st.slider("Frequência de consumo de vegetais/hortaliças", 1.0, 3.0, 2.0, step=0.1, help="1 = Nunca, 2 = Às vezes, 3 = Sempre.")
        ncp = st.slider("Quantidade de refeições principais diárias", 1.0, 4.0, 3.0, step=0.1, help="Número de refeições robustas realizadas por dia.")
        ch2o = st.slider("Consumo diário de água", 1.0, 3.0, 2.0, step=0.1, help="1 = Menos de 1 Litro, 2 = 1 a 2 Litros, 3 = Mais de 2 Litros.")
        faf = st.slider("Frequência de atividade física semanal", 0.0, 3.0, 1.0, step=0.1, help="0 = Nenhuma, 1 = 1 a 2 dias, 2 = 3 a 4 dias, 3 = 5 ou mais dias.")
        tue = st.slider("Tempo diário de uso de telas/tecnologia", 0.0, 2.0, 1.0, step=0.1, help="0 = 0-2h, 1 = 3-5h, 2 = Mais de 5h diárias diante de telas.")
        calc_ui = st.selectbox("Frequência de consumo de álcool", list(map_caec_calc.keys()))

    input_df = pd.DataFrame([{
        'Gender': map_gender[gender_ui], 'Age': age,
        'family_history': map_yes_no[family_history_ui], 'FAVC': map_yes_no[favc_ui], 'FCVC': fcvc, 'NCP': ncp,
        'CAEC': map_caec_calc[caec_ui], 'SMOKE': map_yes_no[smoke_ui], 'CH2O': ch2o, 'SCC': map_yes_no[scc_ui], 'FAF': faf,
        'TUE': tue, 'CALC': map_caec_calc[calc_ui], 'MTRANS': map_mtrans[mtrans_ui]
    }])
    
    if st.button("Calcular Risco de Tendência Epidemiológica"):
        pred_code = pipeline.predict(input_df)[0]
        raw_result = label_mapping[pred_code]
        resultado_pt = map_output.get(raw_result, raw_result)
        
        st.subheader("Tendência de Desenvolvimento Clínico:")
        if "Obesity" in raw_result:
            st.error(f"Alerta Preventivo: Os hábitos e genética atuais correlacionam-se com uma tendência a: **{resultado_pt}**")
        else:
            st.success(f"Prognóstico Estável: Os hábitos atuais indicam tendência a manter-se em: **{resultado_pt}**")

with tab_dash:
    st.header("Visão Estratégica Clínico-Hospitalar")
    try:
        df_raw = pd.read_csv("Obesity.csv")
        df_plot = df_raw.copy()
        df_plot['Obesity'] = df_plot['Obesity'].map(map_output)
        df_plot['family_history'] = df_plot['family_history'].map({'yes': 'Sim', 'no': 'Não'})
        
        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.histogram(
                df_plot, x="family_history", color="Obesity", barmode="group",
                title="Prevalência de Classes por Histórico Familiar (Fator Predominante)",
                labels={"family_history": "Histórico Familiar de Excesso de Peso", "count": "Pacientes", "Obesity": "Nível"}
            )
            st.plotly_chart(fig1, use_container_width=True)
            
        with c2:
            fig2 = px.box(
                df_plot, x="Obesity", y="Age", color="Gender",
                title="Distribuição de Idade por Gênero dentro das Classes Epidemiológicas",
                labels={"Age": "Idade", "Obesity": "Nível de Obesidade", "Gender": "Gênero"}
            )
            st.plotly_chart(fig2, use_container_width=True)
            
        st.markdown("""
        > **Nota de Governança e Segurança (LGPD):** Esta visão analítica consome exclusivamente dados epidemiológicos históricos anonimizados da instituição[cite: 74]. Nenhuma Informação Pessoal Identificável (PII) é armazenada, exposta ou trafegada na nuvem[cite: 99, 100].
        """)
    except FileNotFoundError:
        st.warning("Coloque o arquivo 'Obesity.csv' na pasta raiz para renderizar o dashboard[cite: 74].")