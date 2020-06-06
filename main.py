#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esse programa utiliza de pacotes e módulos que facilitam o acesso a informações disponíveis
na Base de Dados do Materials Project e do Materials Project Contribs.

Objetivo geral : Utilizar Rede Neural para aprender como que a figura de mérito, grandeza que
determina o quão bom termoelétrico é um material, a partir de descritores como Temperatura,
Vetores Posição dos Átomos, Parâmetro de Rede.

Objetivos : 
    - Dados Estruturais : Obtenção de Informações Estruturais do VASP dos 
                          dos materiais que tiveram a análise de transporte (OK)
    - Carrier Transport : Obtenção de Informações dos Portadores de Carga
                          dos Materiais que foram análisado pelo Materials 
                          Project (OK)

    - Organização de Input para a Rede Neural (Descritores) (OK)
    - Organização de Ouput para a Rede Neural (S² sigma , Kappa elétrico) (OK)
    - Criando Banco de Dados com as informações de Input e Output (Em progresso ...)
    - Separação de Dataset para Treino e para Validação
    - Treino da Rede
    - Validação dos resultados 
    - Análise de Resultados e contrução de gráficos de eficiência 

@author: hardproxy
"""
# Importação necessário para o uso do Scikit-Learn

# Caso for necessário :  
# from sklearn.pipeline import make_pipeline
# from sklearn.datasets import load_boston - Dados de Teste da Rede Neural 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


import csv                                            # Módulo de acesso a arquivos do tipo .csv

#from mpcontribs.client import load_client             # Módulo de acesso a uma seção no MPContribs server
#from pymatgen.ext.matproj import MPRester             # Módulo de acesso ao MP server
#from pymatgen import Structure                        # Acesso á classe do tipo Structure

print('Acessando Banco de Dados de Materiais ... :) \n')
# fieldnames = ['','id','identifier','formula','temperature','energy','volume','⟨S²σ⟩p','⟨S²σ⟩n','κₑᵉ']
with open('/home/hardproxy/Documents/TCC-BD_ML/ML-ThermoElec/Data/Materials_Data/Materials_Data.csv') as csv_file:

    csv_reader = csv.DictReader(csv_file)
    
    list_id = []
    list_ident = []
    list_form = []
    list_temp = []
    list_energy = []
    list_volume = [] 
    list_Sp = []
    list_Sn = []
    list_kappa = []
    num_linhas =  0
    for row in csv_reader:
        list_id.append(row['id'])
        list_ident.append(row['identifier'])
        list_form.append(row['formula'])
        list_temp.append(row['temperature'])
        list_energy.append(row['energy'])
        list_volume.append(row['volume'])
        list_Sp.append(row['⟨S²σ⟩p'])
        list_Sn.append(row['⟨S²σ⟩n'])
        list_kappa.append(row['κₑᵉ'])
        num_linhas+=1
csv_file.close()

# Creating dataset
print('Criando Conjunto de Dados ... :) \n')


temp = []

for i in range(14019):
    j = float(list_temp[i])
    temp.append(j)
    
temp_array = np.array(temp)

energy = []

for i in range(14019):
    j = float(list_energy[i])
    energy.append(j)
    
energy_array = np.array(energy)

volume = []

for i in range(14019):
    j = float(list_volume[i])
    volume.append(j)
    
volume_array = np.array(volume)

Sp = []

for i in range(14019):
    j = float(list_Sp[i])
    Sp.append(j)
    
Sp_array = np.array(Sp)

Sn = []

for i in range(14019):
    j = float(list_Sn[i])
    Sn.append(j)
    
Sn_array = np.array(Sn)

kappa = []

for i in range(14019):
    j = float(list_kappa[i])
    kappa.append(j)
    
kappa_array = np.array(kappa)

X = np.c_[temp_array,volume_array,energy_array]
y = np.c_[Sp_array,Sn_array,kappa_array]


# Splitting data ( Train vs Test )
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Starndat Rescaling data 
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train) 
X_test =  scaler.transform(X_test)

# Creating object of Neural Network
ANN  = MLPRegressor(hidden_layer_sizes=(100,100),tol=1e-3, max_iter=10000, random_state=0,verbose=True)

# Training data
ANN.fit(X_train,y_train)

# Testing fase
predictions = ANN.predict(X_test)

score = ANN.score(X_test,y_test)
print("O Score obtido pelo Rede Neural é de : " + str(score) + "\n")
