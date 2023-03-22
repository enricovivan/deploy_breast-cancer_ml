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

        st.title('Breast Cancer - Previsão :D')