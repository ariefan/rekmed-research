from rest_framework.response import Response
from rest_framework import status
from rest_framework import viewsets
from .models import *
from .serializers import *

import gdown
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import mysql.connector
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def clean_html(text):
    soup = BeautifulSoup(text, 'html.parser')
    cleaned_text = soup.get_text(separator=" ")
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text


class GetMethod(viewsets.ModelViewSet):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer

    def list(self, request, *args, **kwargs):
        q = request.GET.get('q')

        # Koneksi ke MySQL
        host = '103.150.190.238'
        user = 'stbi'
        password = 'stbi'
        database = 'rekmed_new'

        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            auth_plugin='mysql_native_password'
        )
        query = """
        SELECT
        p.mr, p.klinik_id, p.nama, p.tanggal_lahir, p.jk, p.alamat, p.no_telp, p.pekerjaan ,
        rm.rm_id, rm.tekanan_darah, rm.nadi, rm.respirasi_rate, rm.suhu,
        rm.berat_badan, rm.tinggi_badan, rm.bmi, rm.assesment, rm.plan, rm.keluhan_utama, rm.anamnesis,
        rm.pemeriksaan_fisik, rm.hasil_penunjang, rm.deskripsi_tindakan, rm.saran_pemeriksaan, rm.alergi_obat,
        GROUP_CONCAT(ro.nama_obat) nama_obat,
        GROUP_CONCAT(CONCAT(rd.kode, ' - ', rd.nama_diagnosis)) diagnosis
        FROM
        pasien p
        LEFT JOIN rekam_medis rm USING(mr)
        LEFT JOIN rm_diagnosis rd USING(rm_id)
        LEFT JOIN rm_obat ro USING(rm_id)
        GROUP BY p.mr, rm_id
        """
        df = pd.read_sql(query, connection)

        # handle missing value in column decribe
        cols_to_fill = ['assesment', 'plan', 'keluhan_utama', 'anamnesis', 'pemeriksaan_fisik', 'hasil_penunjang', 'deskripsi_tindakan', 'saran_pemeriksaan',
                        'alergi_obat', 'diagnosis', 'tekanan_darah', 'nadi', 'respirasi_rate', 'suhu', 'berat_badan', 'tinggi_badan', 'bmi', "rm_id", "nama_obat"]
        for col in cols_to_fill:
            df[col].fillna('', inplace=True)

        # Clean HTML
        html_columns = ['assesment', 'plan', 'keluhan_utama', 'anamnesis',
                        'pemeriksaan_fisik', 'hasil_penunjang', 'deskripsi_tindakan', 'saran_pemeriksaan']
        for column in df.columns:
            if column in html_columns:
                df[column] = df[column].apply(clean_html)

        # Normalisasi
        text_columns = ['nama', 'nama_obat', 'diagnosis']
        for col in text_columns:
            df[col] = df[col].str.lower()  # normalisasi

        # Penghapusan stop word
        from nltk.corpus import stopwords
        nltk.download('stopwords')
        stop_words = set(stopwords.words('indonesian'))
        text_columns_with_stopwords = ['diagnosis', 'nama_obat']
        for col in text_columns_with_stopwords:
            df[col] = df[col].apply(lambda text: ' '.join(
                [word for word in text.split() if word not in stop_words]))

        # Stemming
        stemmer_factory = StemmerFactory()
        stemmer = stemmer_factory.create_stemmer()
        # df['diagnosis'] = df['diagnosis'].apply(lambda text: ' '.join(
        #     [stemmer.stem(word) for word in text.split()]))

        # # Pastikan sudah mengunduh resource yang dibutuhkan dari nltk jika belum
        nltk.download('punkt')

        # Misalkan df adalah DataFrame Anda
        # Tokenisasi kolom 'nama_pasien', 'obat', dan 'diagnosis'
        df['nama_tokenized'] = df['nama'].apply(word_tokenize)
        df['obat_tokenized'] = df['nama_obat'].apply(word_tokenize)
        df['diagnosis_tokenized'] = df['diagnosis'].apply(word_tokenize)

        # Tampilkan hasil tokenisasi
        df[['nama_tokenized', 'obat_tokenized', 'diagnosis_tokenized']].head()

        # Sample data (replace with your dataset)

        documents = df["nama"].astype(str) + " " + df["nama_obat"].astype(str) + " " + df["diagnosis"].astype(str) + " " + df["assesment"].astype(
            str) + " " + df["berat_badan"].astype(str) + " " + df["tinggi_badan"].astype(str) + " " + df["suhu"].astype(str) + " " + df["bmi"].astype(
                str) + " " + df["plan"].astype(str) + " " + df["keluhan_utama"].astype(str) + " " + df["anamnesis"].astype(str) + " " + df["pemeriksaan_fisik"].astype(str)
        query = q

        # Create a TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer()

        # Calculate TF-IDF for the combined "nama," "nama_obat," and "diagnosis" columns
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

        # Calculate TF-IDF for the query
        query_tfidf = tfidf_vectorizer.transform([query])

        # Compute cosine similarities between the query and all documents
        cosine_similarities = linear_kernel(query_tfidf, tfidf_matrix)

        # Rank documents by cosine similarity
        document_scores = list(enumerate(cosine_similarities[0]))
        document_scores.sort(key=lambda x: x[1], reverse=True)

        # Get the top N matching documents
        top_n = 10  # Change to the desired number of top results
        top_matches = document_scores[:top_n]

        results = []

        # Print the top matching documents and their indices
        for idx, score in top_matches:
            row = []
            if score > 0:
                row.append(df['mr'][idx])
                row.append(df['rm_id'][idx])
                row.append(documents[idx])
                results.append(row)
                print(
                    f"Document {idx}: {documents[idx]}, Similarity Score: {score:.4f}")

        return Response(results)

    # def list(self, request, *args, **kwargs):
    #     data = list(Product.objects.all().values())
    #     return Response(data)

    def retrieve(self, request, *args, **kwargs):
        data = list(Product.objects.filter(id=kwargs['pk']).values())
        return Response(data)

    def create(self, request, *args, **kwargs):
        product_serializer_data = ProductSerializer(data=request.data)
        if product_serializer_data.is_valid():
            product_serializer_data.save()
            status_code = status.HTTP_201_CREATED
            return Response({"message": "Product Added Sucessfully", "status": status_code})
        else:
            status_code = status.HTTP_400_BAD_REQUEST
            return Response({"message": "please fill the datails", "status": status_code})

    def destroy(self, request, *args, **kwargs):
        product_data = Product.objects.filter(id=kwargs['pk'])
        if product_data:
            product_data.delete()
            status_code = status.HTTP_201_CREATED
            return Response({"message": "Product delete Sucessfully", "status": status_code})
        else:
            status_code = status.HTTP_400_BAD_REQUEST
            return Response({"message": "Product data not found", "status": status_code})

    def update(self, request, *args, **kwargs):
        product_details = Product.objects.get(id=kwargs['pk'])
        product_serializer_data = ProductSerializer(
            product_details, data=request.data, partial=True)
        if product_serializer_data.is_valid():
            product_serializer_data.save()
            status_code = status.HTTP_201_CREATED
            return Response({"message": "Product Update Sucessfully", "status": status_code})
        else:
            status_code = status.HTTP_400_BAD_REQUEST
            return Response({"message": "Product data Not found", "status": status_code})
