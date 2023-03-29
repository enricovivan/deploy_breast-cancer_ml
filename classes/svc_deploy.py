import streamlit as st
import plotly as pl
import pandas as pd
import numpy as np
import pickle as pkl

import joblib

class SVCDeploy:

    def __init__(self) -> None:

        # Carregando modelo
        # self.modelo = open('./classifiers/svc_bc.pkl', 'rb')
        # self.classifier = pkl.load(self.modelo)

        self.classifier = joblib.load('./classifiers/svc_bc.pkl')

        self.main_window()

    def predicao(self, radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
                radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
                radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst):

        prediction = self.classifier.predict([[
            radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
            radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
            radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst
        ]])
        
        print(prediction)

        return prediction


    def main_window(self):

        st.title('Breast Cancer - Previsão')
        st.text('Para realizarmos a previsão precisaremos coletas os seguintes dados:')
        st.text("""
            radius_mean      texture_mean
            perimeter_mean   area_mean
            smoothness_mean  compactness_mean
            concavity_mean   concave_points_mean
            symmetry_mean    fractal_dimension_mean
            ________________________________________
            radius_se        texture_se
            perimeter_se     area_se
            smoothness_se    compactness_se
            concavity_se     concave_points_se
            symmetry_se      fractal_dimension_se
            ________________________________________
            radius_worst     texture_worst
            perimeter_worst  area_worst
            smoothness_worst compactness_worst
            concavity_worst  concave_points_worst
            symmetry_worst   fractal_dimension_worst
            
        """)

        st.subheader('Digite os Valores:')

        self.col1, self.col2 = st.columns(2)

        # Means
        # self.radius_mean = self.col1.text_input('radius_mean', placeholder='Ex.: 11.23')
        # self.texture_mean = self.col2.text_input('texture_mean', placeholder='Ex.: 20.38')
        # self.perimeter_mean = self.col1.text_input('perimeter_mean', placeholder='Ex.:130.00')
        # self.area_mean = self.col2.text_input('area_mean', placeholder='Ex.: 577.32')
        # self.smoothness_mean = self.col1.text_input('smoothness_mean', placeholder='Ex.: 0.118')
        # self.compactness_mean = self.col2.text_input('compactness_mean', placeholder='Ex.: 0.277')
        # self.concavity_mean = self.col1.text_input('concavity_mean', placeholder='Ex.: 0.3')
        # self.concave_points_mean = self.col2.text_input('concave_points_mean', placeholder='Ex.: 0.05')
        # self.symmetry_mean = self.col1.text_input('symmetry_mean', placeholder='Ex.: 0.2419')
        # self.fractal_dimension_mean = self.col2.text_input('fractal_dimension_mean', placeholder='Ex.: 0.07')

        # # SEs
        # self.radius_se = self.col1.text_input('radius_se', placeholder='Ex.: 0.540')
        # self.texture_se = self.col2.text_input('texture_se', placeholder='Ex.: 0.90')
        # self.perimeter_se = self.col1.text_input('perimeter_se', placeholder='Ex.: 8589')
        # self.area_se = self.col2.text_input('area_se', placeholder='Ex.: 153.4')
        # self.smoothness_se = self.col1.text_input('smoothness_se', placeholder='Ex.: 0.006')
        # self.compactness_se = self.col2.text_input('compactness_se', placeholder='Ex.: 0.04')
        # self.concavity_se = self.col1.text_input('concavity_se', placeholder='Ex.: 0.05')
        # self.concave_points_se = self.col2.text_input('concave_points_se', placeholder='Ex.: 0.01')
        # self.symmetry_se = self.col1.text_input('symmetry_se', placeholder='Ex.: 0.031')
        # self.fractal_dimension_se = self.col2.text_input('fractal_dimension_se', placeholder='Ex.: 0.0061')

        # # Worsts
        # self.radius_worst = self.col1.text_input('radius_worst', placeholder='Ex.: 25.38')
        # self.texture_worst = self.col2.text_input('texture_worst', placeholder='Ex.: 17.33')
        # self.perimeter_worst = self.col1.text_input('perimeter_worst', placeholder='Ex.: 184.3')
        # self.area_worst = self.col2.text_input('area_worst', placeholder='Ex.: 2019')
        # self.smoothness_worst = self.col1.text_input('smoothness_worst', placeholder='Ex.: 0.16')
        # self.compactness_worst = self.col2.text_input('compactness_worst', placeholder='Ex.: 0.675')
        # self.concavity_worst = self.col1.text_input('concavity_worst', placeholder='Ex.: 0.711')
        # self.concave_points_worst = self.col2.text_input('concave_points_worst', placeholder='Ex.: 0.266')
        # self.symmetry_worst = self.col1.text_input('symmetry_worst', placeholder='Ex.: 0.465')
        # self.fractal_dimension_worst = self.col2.text_input('fractal_dimension_worst', placeholder='Ex.: 0.118')

        # Means
        self.radius_mean = self.col1.text_input('radius_mean', placeholder='Ex.: 11.23')
        self.texture_mean = self.col2.text_input('texture_mean', placeholder='Ex.: 20.38')
        self.perimeter_mean = self.col1.text_input('perimeter_mean', placeholder='Ex.:130.00')
        self.area_mean = self.col2.text_input('area_mean', placeholder='Ex.: 577.32')
        self.smoothness_mean = self.col1.text_input('smoothness_mean', placeholder='Ex.: 0.118')
        self.compactness_mean = self.col2.text_input('compactness_mean', placeholder='Ex.: 0.277')
        self.concavity_mean = self.col1.text_input('concavity_mean', placeholder='Ex.: 0.3')
        self.concave_points_mean = self.col2.text_input('concave_points_mean', placeholder='Ex.: 0.05')
        self.symmetry_mean = self.col1.text_input('symmetry_mean', placeholder='Ex.: 0.2419')
        self.fractal_dimension_mean = self.col2.text_input('fractal_dimension_mean', placeholder='Ex.: 0.07')

        # SEs
        self.radius_se = self.col1.text_input('radius_se', placeholder='Ex.: 0.540')
        self.texture_se = self.col2.text_input('texture_se', placeholder='Ex.: 0.90')
        self.perimeter_se = self.col1.text_input('perimeter_se', placeholder='Ex.: 8589')
        self.area_se = self.col2.text_input('area_se', placeholder='Ex.: 153.4')
        self.smoothness_se = self.col1.text_input('smoothness_se', placeholder='Ex.: 0.006')
        self.compactness_se = self.col2.text_input('compactness_se', placeholder='Ex.: 0.04')
        self.concavity_se = self.col1.text_input('concavity_se', placeholder='Ex.: 0.05')
        self.concave_points_se = self.col2.text_input('concave_points_se', placeholder='Ex.: 0.01')
        self.symmetry_se = self.col1.text_input('symmetry_se', placeholder='Ex.: 0.031')
        self.fractal_dimension_se = self.col2.text_input('fractal_dimension_se', placeholder='Ex.: 0.0061')

        # Worsts
        self.radius_worst = self.col1.text_input('radius_worst', placeholder='Ex.: 25.38')
        self.texture_worst = self.col2.text_input('texture_worst', placeholder='Ex.: 17.33')
        self.perimeter_worst = self.col1.text_input('perimeter_worst', placeholder='Ex.: 184.3')
        self.area_worst = self.col2.text_input('area_worst', placeholder='Ex.: 2019')
        self.smoothness_worst = self.col1.text_input('smoothness_worst', placeholder='Ex.: 0.16')
        self.compactness_worst = self.col2.text_input('compactness_worst', placeholder='Ex.: 0.675')
        self.concavity_worst = self.col1.text_input('concavity_worst', placeholder='Ex.: 0.711')
        self.concave_points_worst = self.col2.text_input('concave_points_worst', placeholder='Ex.: 0.266')
        self.symmetry_worst = self.col1.text_input('symmetry_worst', placeholder='Ex.: 0.465')
        self.fractal_dimension_worst = self.col2.text_input('fractal_dimension_worst', placeholder='Ex.: 0.118')

        # Botão para dar predict
        if st.button('Prever'):
            result = self.predicao(self.radius_mean, self.texture_mean, self.perimeter_mean, self.area_mean, self.smoothness_mean, self.compactness_mean, self.concavity_mean, self.concave_points_mean, self.symmetry_mean, self.fractal_dimension_mean,
                                   self.radius_se, self.texture_se, self.perimeter_se, self.area_se, self.smoothness_se, self.compactness_se, self.concavity_se, self.concave_points_se, self.symmetry_se, self.fractal_dimension_se,
                                   self.radius_worst, self.texture_worst, self.perimeter_worst, self.area_worst, self.smoothness_worst, self.compactness_worst, self.concavity_worst, self.concave_points_worst, self.symmetry_worst, self.fractal_dimension_worst)

            if result == 0:
                result = 'Maligno'
            else:
                result = 'Benigno'
            
            st.success('Resultado: {}'.format(result))