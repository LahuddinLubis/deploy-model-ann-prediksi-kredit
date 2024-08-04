import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import layers
from keras import models
from keras import callbacks
from keras.callbacks import History

import sklearn
from sklearn import metrics
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# # loading the saved models# URL file model di GitHub
url = 'https://raw.githubusercontent.com/LahuddinLubis/deploy-model-ann-prediksi-kredit/master/best_ann_model.joblib'
local_filename = 'best_ann_model.joblib'

# # Fungsi untuk mendownload file dari URL
def download_file(url, local_filename):
    response = requests.get(url)
    with open(local_filename, 'wb') as f:
        f.write(response.content)

# # Cek apakah file sudah ada di lokal, jika tidak download
if not os.path.exists(local_filename):
    st.write("Downloading model file...")
    download_file(url, local_filename)

# # Muat model menggunakan joblib
model_prediksi = joblib.load(local_filename)

# # defining the function which will make the prediction using the data which the user inputs 
# def prediction(Nama_Kelompok, Usia, Status_Pernikahan, Pekerjaan, Jumlah_Keluarga, Jumlah_Pinjaman, Jangka_Waktu):
#     # Pre-processing user input
#     if Nama_Kelompok == 'ADIL MAKMUR':
#         Nama_Kelompok = 0
#     elif Nama_Kelompok == 'AN NUR':
#         Nama_Kelompok = 1
#     elif Nama_Kelompok == 'AN-NISA':
#         Nama_Kelompok = 2
#     elif Nama_Kelompok == 'ANGGREK':
#         Nama_Kelompok = 3
#     elif Nama_Kelompok == 'ANUGRAH ADM':
#         Nama_Kelompok = 4
#     elif Nama_Kelompok == 'AS SYIFA 2':
#         Nama_Kelompok = 5
#     elif Nama_Kelompok == 'AS-SYIFA HT II AG':
#         Nama_Kelompok = 6
#     elif Nama_Kelompok == 'AZZAHRA':
#         Nama_Kelompok = 7
#     elif Nama_Kelompok == 'BAHAGIA PENGGALANGAN':
#         Nama_Kelompok = 8
#     elif Nama_Kelompok == 'Barokah Sejahtera':
#         Nama_Kelompok = 9
#     elif Nama_Kelompok == 'BHINEKA HT II UB':    
#         Nama_Kelompok = 10
#     elif Nama_Kelompok == 'Bonsai Mekaar':
#         Nama_Kelompok = 11
#     elif Nama_Kelompok == 'BUBU PENGKOLAN':
#         Nama_Kelompok = 12
#     elif Nama_Kelompok == 'Bunga Mekar':
#         Nama_Kelompok = 13
#     elif Nama_Kelompok == 'CAHAYA MEKAR':
#         Nama_Kelompok = 14
#     elif Nama_Kelompok == 'Cendana':    
#         Nama_Kelompok = 15
#     elif Nama_Kelompok == 'Coklat II Aek Nauli':
#         Nama_Kelompok = 16
#     elif Nama_Kelompok == 'Ganda Pengkolan':
#         Nama_Kelompok = 17
#     elif Nama_Kelompok == 'GARUDA HUTA IRN':
#         Nama_Kelompok = 18
#     elif Nama_Kelompok == 'Cendana':    
#         Nama_Kelompok = 19
#     elif Nama_Kelompok == 'Huta Sidosemi Mekaar':    
#         Nama_Kelompok = 20
#     elif Nama_Kelompok == 'Huta V Bosar Maligas':    
#         Nama_Kelompok = 21
#     elif Nama_Kelompok == 'Huta V Panglong':    
#         Nama_Kelompok = 22
#     elif Nama_Kelompok == 'ISSABELA 2':    
#         Nama_Kelompok = 23
#     elif Nama_Kelompok == 'ISSABELLA PENGKOLAN':    
#         Nama_Kelompok = 24
#     elif Nama_Kelompok == 'Jaya Mekaar':    
#         Nama_Kelompok = 25
#     elif Nama_Kelompok == 'Kampung Baru':    
#         Nama_Kelompok = 26
#     elif Nama_Kelompok == 'KHARISMA MEKAR':    
#         Nama_Kelompok = 27
#     elif Nama_Kelompok == 'KURMA MANIS SL':    
#         Nama_Kelompok = 28
#     elif Nama_Kelompok == 'LESTARI PENGKOLAN':    
#         Nama_Kelompok = 29
#     elif Nama_Kelompok == 'MAMI IDOLA PENGKOLAN':    
#         Nama_Kelompok = 30
#     elif Nama_Kelompok == 'Mangga Mekaar':    
#         Nama_Kelompok = 31
#     elif Nama_Kelompok == 'MATINGGI MEKAR':    
#         Nama_Kelompok = 32
#     elif Nama_Kelompok == 'MEKAAR AFD II TELADAN':    
#         Nama_Kelompok = 33
#     elif Nama_Kelompok == 'Mekaar AL-Awal':    
#         Nama_Kelompok = 34
#     elif Nama_Kelompok == 'Mekaar Aman Tinjowan':    
#         Nama_Kelompok = 35
#     elif Nama_Kelompok == 'Mekaar Baik Riahnaposo':    
#         Nama_Kelompok = 36
#     elif Nama_Kelompok == 'Mekaar Emas Sidosemi':    
#         Nama_Kelompok = 37
#     elif Nama_Kelompok == 'Mekaar Gunung Bayu':    
#         Nama_Kelompok = 38
#     elif Nama_Kelompok == 'Cendana':    
#         Nama_Kelompok = 39
#     elif Nama_Kelompok == 'MEKAAR JAYA ADM':    
#         Nama_Kelompok = 40
#     elif Nama_Kelompok == 'MEKAAR JINGGAH ADM':    
#         Nama_Kelompok = 41
#     elif Nama_Kelompok == 'MEKAAR LAPIAN':    
#         Nama_Kelompok = 42
#     elif Nama_Kelompok == 'MEKAAR PULO PITU MARIHAT':    
#         Nama_Kelompok = 43
#     elif Nama_Kelompok == 'Mekaar Sukarejo':    
#         Nama_Kelompok = 44
#     elif Nama_Kelompok == 'MEKAAR TALUN JAYA':    
#         Nama_Kelompok = 45
#     elif Nama_Kelompok == 'MEKAR MULIA':    
#         Nama_Kelompok = 46
#     elif Nama_Kelompok == 'MELATI':    
#         Nama_Kelompok = 47
#     elif Nama_Kelompok == 'NUSA INDAH':    
#         Nama_Kelompok = 48
#     elif Nama_Kelompok == 'Padang Matinggi':    
#         Nama_Kelompok = 49
#     elif Nama_Kelompok == 'PDM MEKAAR':    
#         Nama_Kelompok = 50
#     elif Nama_Kelompok == 'Penggalangan Mekaar':    
#         Nama_Kelompok = 51
#     elif Nama_Kelompok == 'Pinang Mekaar':    
#         Nama_Kelompok = 52
#     elif Nama_Kelompok == 'PISANG SIDOSEMI':    
#         Nama_Kelompok = 53
#     elif Nama_Kelompok == 'PONDOK GEREJA':    
#         Nama_Kelompok = 54    
#     elif Nama_Kelompok == 'Pondok Kolam PB':    
#         Nama_Kelompok = 55
#     elif Nama_Kelompok == 'RAMBUTAN ADM':    
#         Nama_Kelompok = 56
#     elif Nama_Kelompok == 'SEJAHTERA HT III TBD':    
#         Nama_Kelompok = 57
#     elif Nama_Kelompok == 'SITIO TIO ADM':    
#         Nama_Kelompok = 58
#     elif Nama_Kelompok == 'Ujung Bayu':    
#         Nama_Kelompok = 59    
#     elif Nama_Kelompok == 'Ujung Padang':    
#         Nama_Kelompok = 60

#     Usia = Usia

#     if Status_Pernikahan == 'Belum Menikah':
#         Status_Pernikahan = 0
#     elif Status_Pernikahan == 'Menikah':
#         Status_Pernikahan = 1
    
#     if Pekerjaan == 'Pedagang':
#         Pekerjaan = 0
#     elif Pekerjaan == 'Wiraswasta':
#         Pekerjaan = 1

#     Jumlah_Keluarga = Jumlah_Keluarga
#     Jumlah_Pinjaman = Jumlah_Pinjaman
#     Jangka_Waktu = Jangka_Waktu

# #     # # Making predictions 
#     prediction = ""
#     prediction = model_prediksi.predict_proba([[Nama_Kelompok, Usia, Status_Pernikahan, Pekerjaan,
#                                                 Jumlah_Keluarga, Jumlah_Pinjaman, Jangka_Waktu]])

#     if (prediction[1] >= 0.5):
#         st.error("Hasil prediksi status pinjaman : Macet :thumbsdown:")
#         # st.image('images/rejected.jpg', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
#     else:
#         st.success("Hasil prediksi status pinjaman : Lancar :thumbsup:")
#         # st.image('images/approved.jpg', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
#     return prediction

def app():
    st.subheader('Prediksi Pembayaran Pinjaman Nasabah PNM Mekaar')
    # col1, col2 = st.columns(2)
    # with col1:
    #     Nama_Kelompok = st.selectbox('Nama Kelompok',("ADIL MAKMUR", "AN NUR", "AN-NISA", "ANGGREK", "ANUGRAH ADM", "AS SYIFA 2", "AS-SYIFA HT II AG", "AZZAHRA",
    #                                      "BAHAGIA PENGGALANGAN", "Barokah Sejahtera", "BHINEKA HT II UB", "Bonsai Mekaar", "BUBU PENGKOLAN", "Bunga Mekar",
    #                                      "CAHAYA MEKAR", "Cendana", "Coklat II Aek Nauli", "Ganda Pengkolan", "GARUDA HUTA IRN", "Huta Sidosemi Mekaar",
    #                                      "Huta V Bosar Maligas", "Huta V Panglong", "ISSABELA 2", "ISSABELLA PENGKOLAN", "Jaya Mekaar", "Kampung Baru",
    #                                      "KHARISMA MEKAR", "KURMA MANIS SL", "LESTARI PENGKOLAN", "MAMI IDOLA PENGKOLAN", "Mangga Mekaar", "MATINGGI MEKAR",
    #                                      "MEKAAR AFD II TELADAN", "Mekaar AL-Awal", "Mekaar Aman Tinjowan", "Mekaar Baik Riahnaposo", "Mekaar Emas Sidosemi",
    #                                      "Mekaar Gunung Bayu", "MEKAAR JAYA ADM", "MEKAAR JINGGAH ADM", "MEKAAR LAPIAN", "MEKAAR PULO PITU MARIHAT",
    #                                      "Mekaar Sukarejo", "MEKAAR TALUN JAYA", "MEKAR MULIA", "MELATI", "NUSA INDAH", "Padang Matinggi", "PDM MEKAAR",
    #                                      "Penggalangan Mekaar", "Pinang Mekaar", "PISANG SIDOSEMI", "PONDOK GEREJA", "Pondok Kolam PB", "RAMBUTAN ADM",
    #                                      "SEJAHTERA HT III TBD", "SITIO TIO ADM", "Ujung Bayu", "Ujung Padang"))
    #     Status_Pernikahan = st.radio("Status Pernikahan", ('Belum Menikah', 'Menikah'))
    #     Jumlah_Keluarga = st.number_input("Jumlah Keluarga", 1,10)       
    # with col2:
    #     Usia = st.number_input("Masukkan Usia", 22,70)
    #     Pekerjaan = st.radio("Pekerjaan", ['Pedagang','Wiraswasta'])
    #     Jangka_Waktu = st.number_input("Jangka Waktu", 12,36)                
    # Jumlah_Pinjaman = st.number_input("Jumlah Pinjaman", 1000000,10000000)
    

    # # when 'Predict' is clicked, make the prediction and store it
    # if st.button(":dollar: Prediksi"):        
    #     hasil_prediksi = prediction(Nama_Kelompok, Usia, Status_Pernikahan, Pekerjaan,
    #                                 Jumlah_Keluarga, Jumlah_Pinjaman, Jangka_Waktu)
    #     hasil_prediksi
