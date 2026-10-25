r"""
recommender.py
==============
Sistema de Recomendação baseado em Filtragem Colaborativa Item-Item.
Implementa a Correlação de Pearson entre filmes e a Regressão com K-Nearest Neighbors (KNN).

Fórmulas Implementadas:
-----------------------
1. Similaridade de Pearson entre filmes i e j:
   $$c(i,j) = \frac{\sum_{k=1}^{n} (r_i^k - \bar{r}_i)(r_j^k - \bar{r}_j)}
   {\sqrt{\sum_{k=1}^{n} (r_i^k - \bar{r}_i)^2} \sqrt{\sum_{k=1}^{n} (r_j^k - \bar{r}_j)^2}}$$

2. Inferência de Nota (Regressão KNN):
   $$\hat{r}^u_i = \frac{\sum_{j \in \mathcal{N}_k(i) \cap I_u} c(i,j) \cdot r_j^u}
   {\sum_{j \in \mathcal{N}_k(i) \cap I_u} |c(i,j)|}$$
"""

import time
import numpy as np
import pandas as pd
from scipy import sparse
from typing import Dict, List, Tuple, Any, Optional


class ItemItemKNNRecommender:
    """
    Recomendador baseado em filtragem colaborativa item-item usando
    correlação de Pearson e regressão KNN com otimização matricial.
    """

    def __init__(self, k_neighbors: int = 20, min_common_users: int = 2):
        self.k_neighbors = k_neighbors
        self.min_common_users = min_common_users

        # Estruturas de dados treinadas
        self.movie_means: Dict[int, float] = {}
        self.user_means: Dict[int, float] = {}
        self.global_mean: float = 3.5

        # Mapeamentos de ID para índices
        self.users: List[int] = []
        self.movies: List[int] = []
        self.u_map: Dict[int, int] = {}
        self.m_map: Dict[int, int] = {}

        # Matrizes esparsas para cálculo eficiente
        self.mat_norm: Optional[sparse.csr_matrix] = None
        self.bin_mat: Optional[sparse.csr_matrix] = None
        self.user_ratings: Dict[int, Dict[int, float]] = {}
        self.movie_ratings: Dict[int, Dict[int, float]] = {}
        self.movie_titles: Dict[int, str] = {}

    def fit(self, train_df: pd.DataFrame, movies_df: Optional[pd.DataFrame] = None):
        """
        Treina o recomendador calculando médias e normalizações centradas nos itens.
        """
        print("[Recommender] Ajustando modelo com dados de treino...")
        t0 = time.time()

        if movies_df is not None:
            self.movie_titles = dict(zip(movies_df["movieId"], movies_df["title"]))

        self.global_mean = float(train_df["rating"].mean())
        self.movie_means = train_df.groupby("movieId")["rating"].mean().to_dict()
        self.user_means = train_df.groupby("userId")["rating"].mean().to_dict()

        # Dicionários rápidos de busca
        self.user_ratings = train_df.groupby("userId").apply(
            lambda df: dict(zip(df["movieId"], df["rating"])), include_groups=False
        ).to_dict()

        self.movie_ratings = train_df.groupby("movieId").apply(
            lambda df: dict(zip(df["userId"], df["rating"])), include_groups=False
        ).to_dict()

        # Criação de índices matriciais
        self.users = sorted(train_df["userId"].unique())
        self.movies = sorted(train_df["movieId"].unique())
        self.u_map = {u: idx for idx, u in enumerate(self.users)}
        self.m_map = {m: idx for idx, m in enumerate(self.movies)}

        u_idx = train_df["userId"].map(self.u_map).values
        m_idx = train_df["movieId"].map(self.m_map).values
        r_vals = train_df["rating"].values

        # Subtrai a média de cada filme: (r_i^k - \bar{r}_i)
        m_mean_arr = np.array([self.movie_means[m] for m in self.movies])
        centered_r = r_vals - m_mean_arr[m_idx]

        # Matriz esparsa de itens x usuários com notas centradas
        n_m = len(self.movies)
        n_u = len(self.users)
        mat = sparse.csr_matrix((centered_r, (m_idx, u_idx)), shape=(n_m, n_u))

        # Norma L2 de cada linha para o cálculo de cosseno/Pearson
        row_norms = np.sqrt(mat.multiply(mat).sum(axis=1)).A1
        row_norms[row_norms == 0.0] = 1.0

        # Matriz normalizada: dot product entre linhas fornece a correlação de Pearson exata
        self.mat_norm = mat.multiply(sparse.csr_matrix(1.0 / row_norms[:, None])).tocsr()

        # Matriz binária para contagem de usuários em comum (co-raters)
        self.bin_mat = sparse.csr_matrix((np.ones_like(r_vals), (m_idx, u_idx)), shape=(n_m, n_u))

        print(f"[Recommender] Treinamento concluído em {time.time()-t0:.2f}s. "
              f"Filmes={len(self.movies):,}, Usuários={len(self.users):,}")

    def pearson_similarity(self, movie1: int, movie2: int) -> Tuple[float, int]:
        """
        Calcula a correlação de Pearson direta entre dois filmes conforme a fórmula:
        Retorna (correlação, quantidade_de_co_raters).
        """
        if movie1 not in self.movie_ratings or movie2 not in self.movie_ratings:
            return 0.0, 0

        u1 = self.movie_ratings[movie1]
        u2 = self.movie_ratings[movie2]
        common = set(u1.keys()).intersection(u2.keys())
        co_raters = len(common)

        if co_raters < self.min_common_users:
            return 0.0, co_raters

        mean1 = self.movie_means[movie1]
        mean2 = self.movie_means[movie2]

        num = 0.0
        den1 = 0.0
        den2 = 0.0
        for u in common:
            diff1 = u1[u] - mean1
            diff2 = u2[u] - mean2
            num += diff1 * diff2
            den1 += diff1 ** 2
            den2 += diff2 ** 2

        if den1 == 0.0 or den2 == 0.0:
            return 0.0, co_raters

        sim = num / (np.sqrt(den1) * np.sqrt(den2))
        return float(sim), co_raters

    def extract_similarity_edges(self, max_edges_per_movie: int = 10,
                                 min_similarity: float = 0.25,
                                 min_co_raters: int = 3,
                                 max_total_edges: int = 20000) -> List[Dict[str, Any]]:
        """
        Extrai as arestas de similaridade mais relevantes entre filmes para povoar o Grafo
        (Neo4j ou NetworkX) com relacionamentos (:Movie)-[:SIMILAR]->(:Movie).
        """
        print("[Recommender] Extraindo arestas de similaridade (:SIMILAR) para o grafo...")
        edges = []
        seen_pairs = set()

        for m1 in self.movies:
            m1_idx = self.m_map[m1]
            # Calcula similaridade do filme m1 com todos os outros filmes
            sim_row = self.mat_norm[m1_idx].dot(self.mat_norm.T).toarray().ravel()
            co_row = self.bin_mat[m1_idx].dot(self.bin_mat.T).toarray().ravel()

            # Filtrar auto-similaridade, similaridade mínima e co-avaliadores mínimos
            valid_mask = (sim_row >= min_similarity) & (co_row >= min_co_raters)
            valid_mask[m1_idx] = False

            if not np.any(valid_mask):
                continue

            valid_indices = np.where(valid_mask)[0]
            valid_sims = sim_row[valid_indices]
            valid_cos = co_row[valid_indices]

            # Ordenar pelos maiores
            sort_order = np.argsort(valid_sims)[::-1][:max_edges_per_movie]

            for s_idx in sort_order:
                m2_idx = valid_indices[s_idx]
                m2 = self.movies[m2_idx]
                pair = (min(m1, m2), max(m1, m2))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    edges.append({
                        "movie1": m1,
                        "movie2": m2,
                        "similarity": round(float(valid_sims[s_idx]), 4),
                        "co_raters": int(valid_cos[s_idx])
                    })

            if len(edges) >= max_total_edges:
                break

        print(f"[Recommender] {len(edges):,} arestas de similaridade extraídas.")
        return edges

    def predict_rating(self, user_id: int, movie_id: int, k: Optional[int] = None) -> float:
        r"""
        Infere a nota que o usuário u provavelmente daria ao filme i:
        $$\hat{r}^u_i = \frac{\sum_{j \in \mathcal{N}_k(i) \cap I_u} c(i,j) \cdot r_j^u}{\sum_{j \in \mathcal{N}_k(i) \cap I_u} |c(i,j)|}$$
        """
        if k is None:
            k = self.k_neighbors

        # Se o filme não foi visto no treino, usa média do usuário ou média global
        if movie_id not in self.m_map:
            return self.user_means.get(user_id, self.global_mean)

        m_i = self.m_map[movie_id]
        u_ratings = self.user_ratings.get(user_id, {})
        if not u_ratings:
            return self.movie_means.get(movie_id, self.global_mean)

        # Itens avaliados pelo usuário no treino que estão no catálogo
        u_m_indices = [self.m_map[mj] for mj in u_ratings.keys() if mj in self.m_map and mj != movie_id]
        if not u_m_indices:
            return self.movie_means.get(movie_id, self.global_mean)

        # Similaridade entre movie_id e apenas os filmes avaliados pelo usuário
        sims = self.mat_norm[m_i].dot(self.mat_norm[u_m_indices].T).toarray().ravel()

        # Filtrar similaridades estritamente positivas para a média ponderada
        pos_mask = sims > 0
        if not np.any(pos_mask):
            return self.movie_means.get(movie_id, self.global_mean)

        sims_pos = sims[pos_mask]
        u_m_filtered = np.array(u_m_indices)[pos_mask]

        # Selecionar os K vizinhos mais próximos
        if len(sims_pos) > k:
            top_local_idx = np.argpartition(sims_pos, -k)[-k:]
            top_sims = sims_pos[top_local_idx]
            top_m_indices = u_m_filtered[top_local_idx]
        else:
            top_sims = sims_pos
            top_m_indices = u_m_filtered

        # Notas dadas pelo usuário aos vizinhos
        top_ratings = np.array([u_ratings[self.movies[idx]] for idx in top_m_indices])

        den = np.sum(np.abs(top_sims))
        if den == 0.0:
            return self.movie_means.get(movie_id, self.global_mean)

        pred = float(np.sum(top_sims * top_ratings) / den)
        return float(np.clip(pred, 0.5, 5.0))

    def predict_batch(self, test_df: pd.DataFrame, k: Optional[int] = None) -> np.ndarray:
        """
        Executa predições para todo o DataFrame de teste com otimização matricial.
        """
        if k is None:
            k = self.k_neighbors

        print(f"[Recommender] Gerando predições para {len(test_df):,} avaliações de teste (K={k})...")
        t0 = time.time()
        predictions = []

        for _, row in test_df.iterrows():
            u = int(row["userId"])
            m = int(row["movieId"])
            pred = self.predict_rating(u, m, k=k)
            predictions.append(pred)

        pred_arr = np.array(predictions)
        print(f"[Recommender] Predições finalizadas em {time.time()-t0:.2f}s.")
        return pred_arr

    def recommend(self, user_id: int, top_n: int = 10, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Gera uma lista com os top_n filmes recomendados para um determinado usuário,
        avaliando os candidatos que ele ainda não assistiu.
        """
        if k is None:
            k = self.k_neighbors

        user_seen = set(self.user_ratings.get(user_id, {}).keys())

        # Selecionar candidatos: filmes populares com pelo menos 15 avaliações que o usuário não viu
        candidates = [m for m in self.movies if m not in user_seen and len(self.movie_ratings[m]) >= 15]

        scores = []
        for m in candidates:
            score = self.predict_rating(user_id, m, k=k)
            scores.append((m, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_scores = scores[:top_n]

        results = []
        for rank, (m_id, pred_score) in enumerate(top_scores, 1):
            results.append({
                "rank": rank,
                "movieId": m_id,
                "title": self.movie_titles.get(m_id, f"Movie {m_id}"),
                "predicted_rating": round(pred_score, 2),
                "avg_rating": round(self.movie_means.get(m_id, 0.0), 2)
            })

        return results
