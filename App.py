import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array

# --- Configuração da Página (Visual Médico) ---
st.set_page_config(page_title="Dr. AI Mobile", layout="centered")

st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    h1 {color: #2c3e50; text-align: center;}
    .stButton>button {background-color: #008CBA; color: white; border-radius: 12px; height: 50px; width: 100%;}
    .resultado-box {background-color: #ffffff; padding: 20px; border-radius: 10px; border-left: 5px solid #008CBA; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);}
    </style>
    """, unsafe_allow_html=True)

st.title("🩺 Diagnóstico Visual IA")
st.write("Vovô diz: Envie uma foto (planta, animal, objeto) e a IA dirá o que é!")

# --- Carregando o Cérebro da IA (Cache para não travar) ---
@st.cache_resource
def carregar_modelo():
    # Baixa o modelo MobileNetV2 (leve e rápido para celular)
    model = MobileNetV2(weights='imagenet')
    return model

with st.spinner('Carregando o cérebro digital... (aguarde um pouquinho)'):
    modelo = carregar_modelo()

# --- Upload da Imagem ---
arquivo = st.file_uploader("📸 Tire uma foto ou escolha da galeria", type=['jpg', 'jpeg', 'png'])

if arquivo is not None:
    # 1. Mostrar a imagem carregada
    image = Image.open(arquivo)
    st.image(image, caption='Imagem Analisada', use_column_width=True)
    
    # 2. Botão para iniciar o diagnóstico
    if st.button("🔍 Analisar Imagem Agora"):
        try:
            with st.spinner('A IA está pensando... 🧠'):
                # --- Pré-processamento Cirúrgico ---
                # A IA precisa da imagem em 224x224 pixels exatos
                img_resized = image.resize((224, 224))
                
                # Converter para array numérico
                img_array = img_to_array(img_resized)
                
                # Criar um lote (batch) de 1 imagem
                img_batch = np.expand_dims(img_array, axis=0)
                
                # Ajustar cores para o padrão que a IA entende
                img_preprocessed = preprocess_input(img_batch)
                
                # --- Previsão ---
                prediction = modelo.predict(img_preprocessed)
                
                # Decodificar o resultado (Top 3 chances)
                resultados = decode_predictions(prediction, top=3)[0]

            # --- Exibir Resultados ---
            st.markdown("---")
            st.subheader("📊 Resultado da Análise:")
            
            for i, (id_imagem, label, probabilidade) in enumerate(resultados):
                # Traduzindo visualmente a confiança em porcentagem
                confianca = probabilidade * 100
                
                st.markdown(f"""
                <div class="resultado-box">
                    <h3>#{i+1}: {label.replace('_', ' ').upper()}</h3>
                    <p>Probabilidade: <strong>{confianca:.2f}%</strong></p>
                    <progress value="{int(confianca)}" max="100" style="width:100%"></progress>
                </div>
                <br>
                """, unsafe_allow_html=True)
                
            st.success("Análise concluída com sucesso! ✅")
            st.info("Nota: Para diagnósticos médicos reais, a IA precisaria ser treinada com dados hospitalares específicos. Este é um modelo de demonstração geral.")

        except Exception as e:
            st.error(f"Ops! Algo deu errado na análise: {e}")

else:
    st.info("☝️ Aguardando envio da imagem...")

# Rodapé
st.markdown("---")
st.caption("Desenvolvido no Celular | Tecnologia TensorFlow & Streamlit")
