import streamlit as st
from fastai.vision.all import *
import pathlib, platform
import plotly.express as px
plt = platform.system()
if plt == "Linux": pathlib.WindowsPath = pathlib.PosixPath
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Gullar, Telefonlar, Quchlarni klassifikatsiya qiluvchi model")

file = st.file_uploader(label="Rasm yuklash", type=["Jpg", "Png", "Svg","Gif"])

if file:
    img = PILImage.create(file)
    st.image(img)

    model = load_learner("classification_model.pkl")
    pred, pred_id, probs = model.predict(img)

    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}")    

    fig = plt.figure(figsize=(10,4))
    sns.barplot(x=probs*100, y=model.dls.vocab)
    plt.xlabel("Ehtimollik")
    st.pyplot(fig)

    st.text("Ishlab chiqaruvchi: Saydullayev Asadbek")