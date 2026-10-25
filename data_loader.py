"""
data_loader.py
==============
Módulo responsável por obter, carregar e dividir os dados do MovieLens.
Suporta tanto MovieLens Small (ratings.csv / movies.csv) quanto MovieLens 20M (rating.csv / movie.csv).
"""

import os
import ssl
import zipfile
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATASET_NAME = "ml-latest-small"


def download_and_extract_movielens(data_dir=DEFAULT_DATA_DIR, dataset_name=DATASET_NAME):
    """
    Baixa e extrai o dataset MovieLens caso os arquivos não existam localmente.
    """
    target_dir = os.path.join(data_dir, dataset_name)
    movies_path = os.path.join(target_dir, "movies.csv")
    ratings_path = os.path.join(target_dir, "ratings.csv")

    if os.path.exists(movies_path) and os.path.exists(ratings_path):
        return target_dir

    os.makedirs(data_dir, exist_ok=True)
    zip_dest = os.path.join(data_dir, f"{dataset_name}.zip")

    print(f"[DataLoader] Baixando MovieLens de: {MOVIELENS_URL}...")
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    with urllib.request.urlopen(MOVIELENS_URL, context=ctx) as response, open(zip_dest, "wb") as out_file:
        out_file.write(response.read())

    print(f"[DataLoader] Extraindo arquivo {zip_dest}...")
    with zipfile.ZipFile(zip_dest, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    print(f"[DataLoader] Concluído com sucesso. Arquivos em: {target_dir}")
    return target_dir


def load_data(data_dir=DEFAULT_DATA_DIR, dataset_name=DATASET_NAME, sample_size=None):
    """
    Carrega os dados de filmes e avaliações.
    Detecta automaticamente arquivos nos caminhos:
    - data/ml-latest-small/ (movies.csv, ratings.csv)
    - Raiz do projeto (movie.csv, rating.csv ou movies.csv, ratings.csv)
    """
    root_dir = os.path.dirname(__file__)

    # Prioridade 1: data/ml-latest-small/
    target_dir = os.path.join(data_dir, dataset_name)
    movies_path = os.path.join(target_dir, "movies.csv")
    ratings_path = os.path.join(target_dir, "ratings.csv")

    # Prioridade 2: Raiz com movies.csv / ratings.csv
    if not (os.path.exists(movies_path) and os.path.exists(ratings_path)):
        alt_movies = os.path.join(root_dir, "movies.csv")
        alt_ratings = os.path.join(root_dir, "ratings.csv")
        if os.path.exists(alt_movies) and os.path.exists(alt_ratings):
            movies_path, ratings_path = alt_movies, alt_ratings

    # Prioridade 3: Raiz com movie.csv / rating.csv (ex: MovieLens 20M)
    if not (os.path.exists(movies_path) and os.path.exists(ratings_path)):
        kaggle_movies = os.path.join(root_dir, "movie.csv")
        kaggle_ratings = os.path.join(root_dir, "rating.csv")
        if os.path.exists(kaggle_movies) and os.path.exists(kaggle_ratings):
            movies_path, ratings_path = kaggle_movies, kaggle_ratings

    # Se ainda não encontrou, faz download do ml-latest-small
    if not (os.path.exists(movies_path) and os.path.exists(ratings_path)):
        download_and_extract_movielens(data_dir, dataset_name)
        movies_path = os.path.join(target_dir, "movies.csv")
        ratings_path = os.path.join(target_dir, "ratings.csv")

    print(f"[DataLoader] Lendo filmes de: {movies_path}")
    print(f"[DataLoader] Lendo avaliações de: {ratings_path}")

    movies_df = pd.read_csv(movies_path)
    
    # Se for o dataset de 20M e o usuário solicitar sample, carrega subset para velocidade
    if sample_size is not None:
        ratings_df = pd.read_csv(ratings_path, nrows=sample_size)
    else:
        ratings_df = pd.read_csv(ratings_path)

    # Padronização de nomes de colunas caso venha com diferenças sutis
    movies_df.rename(columns={"movieId": "movieId", "title": "title", "genres": "genres"}, inplace=True)
    ratings_df.rename(columns={"userId": "userId", "movieId": "movieId", "rating": "rating", "timestamp": "timestamp"}, inplace=True)

    # Tratar gêneros (lista de strings)
    movies_df["genres_list"] = movies_df["genres"].apply(
        lambda g: g.split("|") if isinstance(g, str) and g != "(no genres listed)" else []
    )

    print(f"[DataLoader] Filmes carregados: {len(movies_df):,}")
    print(f"[DataLoader] Avaliações carregadas: {len(ratings_df):,}")
    print(f"[DataLoader] Usuários únicos: {ratings_df['userId'].nunique():,}")
    print(f"[DataLoader] Filmes avaliados: {ratings_df['movieId'].nunique():,}")

    return movies_df, ratings_df


def split_data(ratings_df, test_size=0.2, random_state=42):
    """
    Divide as avaliações em 80% treino e 20% teste.
    """
    train_df, test_df = train_test_split(
        ratings_df, test_size=test_size, random_state=random_state
    )
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    train_df["split"] = "train"
    test_df["split"] = "test"

    print(f"[DataLoader] Split concluído: Treino={len(train_df):,} ({(1-test_size)*100:.0f}%), "
          f"Teste={len(test_df):,} ({test_size*100:.0f}%)")
    return train_df, test_df


if __name__ == "__main__":
    movies, ratings = load_data()
    train, test = split_data(ratings)

