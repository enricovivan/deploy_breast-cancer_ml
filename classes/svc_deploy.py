import streamlit as st
import plotly as pl
import pandas as pd
import numpy as np
import pickle

class SVCDeploy:

    def __init__(self) -> None:
        
        # Carregando modelo
        self.modelo = open('./classifiers/svc_bc.pkl', 'rb')
        self.classifier = pickle.load(self.modelo)

        self.main_window()


    def main_window(self):

        st.title('Breast Cancer - Previsão')
        st.text('Para realizarmos a previsão precisaremos coletas os seguintes dados:')
        st.text("""
            radius_mean      texture_mean
            perimeter_mean   area_mean
            smoothness_mean  compactness_mean
            concavity_mean   concave_points_mean
            symmetry_mean    fractal_dimension_mean
            radius_se        texture_se
            perimeter_se     area_se
            smoothness_se    compactness_se
            concavity_se     concave points_se
            symmetry_se      fractal_dimension_se
            radius_worst     texture_worst
            perimeter_worst  area_worst
            smoothness_worst compactness_worst
            concavity_worst  concave points_worst
            symmetry_worst   fractal_dimension_worst
        """)

        st.subheader('Digite os Valores:')

        # Means
        self.radius_mean = st.text_input('radius_mean', placeholder='Ex.: 11.23')
        self.texture_mean = st.text_input('texture_mean', placeholder='Ex.: 20.38')
        self.perimeter_mean = st.text_input('perimeter_mean', placeholder='Ex.:130.00')
        self.area_mean = st.text_input('area_mean', placeholder='Ex.: 577.32')
        self.smoothness_mean = st.text_input('smoothness_mean', placeholder='Ex.: 0.118')
        self.compactness_mean = st.text_input('compactness_mean', placeholder='Ex.: 0.277')
        self.concavity_mean = st.text_input('concavity_mean', placeholder='Ex.: 0.3')
        self.concave_points_mean = st.text_input('concave_points_mean', placeholder='Ex.: 0.05')
        self.symmetry_mean = st.text_input('symmetry_mean', placeholder='Ex.: 0.2419')
        self.fractal_dimension_mean = st.text_input('fractal_dimension_mean', placeholder='Ex.: 0.07')