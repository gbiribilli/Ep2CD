"""
graph_model.py
==============
Modelagem formal do Grafo de Conhecimento e Recomendação do MovieLens.

Definição dos Nós e Arestas:
----------------------------
1. Nós (Nodes):
   - User:
     * Label: :User
     * Atributos: userId (int)
   - Movie:
     * Label: :Movie
     * Atributos: movieId (int), title (str), genres (list[str])
   - Genre:
     * Label: :Genre
     * Atributos: name (str)

2. Arestas (Relationships):
   - RATED:
     * De: (:User) Para: (:Movie)
     * Atributos: rating (float), timestamp (int), split (str: 'train' | 'test')
   - OF_GENRE:
     * De: (:Movie) Para: (:Genre)
     * Atributos: nenhum
   - SIMILAR:
     * De: (:Movie) Para: (:Movie)
     * Atributos: similarity (float, Correlação de Pearson), co_raters (int)
"""

import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional


class MovieGraph:
    """
    Grafo em memória implementado com NetworkX para modelar usuários,
    filmes, gêneros, avaliações e arestas de similaridade calculadas.
    """

    def __init__(self):
        self.g = nx.DiGraph()
        self.movies_lookup: Dict[int, Dict[str, Any]] = {}

    def build_from_dfs(self, movies_df, ratings_df):
        """
        Popula o grafo a partir dos DataFrames de filmes e avaliações.
        """
        print("[MovieGraph] Construindo nós de Filmes e Gêneros...")
        for _, row in movies_df.iterrows():
            m_id = int(row["movieId"])
            title = str(row["title"])
            genres_list = row["genres_list"] if isinstance(row["genres_list"], list) else []

            movie_node = f"m_{m_id}"
            self.g.add_node(
                movie_node,
                node_type="Movie",
                movieId=m_id,
                title=title,
                genres=genres_list
            )
            self.movies_lookup[m_id] = {
                "movieId": m_id,
                "title": title,
                "genres": genres_list
            }

            # Nós e arestas de Gênero (:Movie)-[:OF_GENRE]->(:Genre)
            for g_name in genres_list:
                g_node = f"g_{g_name}"
                if not self.g.has_node(g_node):
                    self.g.add_node(g_node, node_type="Genre", name=g_name)
                self.g.add_edge(movie_node, g_node, rel_type="OF_GENRE")

        print("[MovieGraph] Construindo nós de Usuários e Arestas de Avaliação (RATED)...")
        for _, row in ratings_df.iterrows():
            u_id = int(row["userId"])
            m_id = int(row["movieId"])
            rating = float(row["rating"])
            ts = int(row["timestamp"])
            split = str(row.get("split", "train"))

            user_node = f"u_{u_id}"
            if not self.g.has_node(user_node):
                self.g.add_node(user_node, node_type="User", userId=u_id)

            movie_node = f"m_{m_id}"
            # Se o filme não existir no catálogo, cria nó básico
            if not self.g.has_node(movie_node):
                self.g.add_node(movie_node, node_type="Movie", movieId=m_id, title=f"Movie {m_id}", genres=[])
                self.movies_lookup[m_id] = {"movieId": m_id, "title": f"Movie {m_id}", "genres": []}

            # Aresta de Avaliação (:User)-[:RATED]->(:Movie)
            self.g.add_edge(
                user_node,
                movie_node,
                rel_type="RATED",
                rating=rating,
                timestamp=ts,
                split=split
            )

        print(f"[MovieGraph] Grafo criado com sucesso! "
              f"Nós: {self.g.number_of_nodes():,}, Arestas: {self.g.number_of_edges():,}")

    def add_similarity_edges(self, similarity_edges: List[Dict[str, Any]]):
        """
        Adiciona arestas de similaridade entre filmes no grafo (:Movie)-[:SIMILAR]->(:Movie).
        Cada registro contém: movie1, movie2, similarity, co_raters.
        """
        added = 0
        for edge in similarity_edges:
            m1_node = f"m_{edge['movie1']}"
            m2_node = f"m_{edge['movie2']}"
            if self.g.has_node(m1_node) and self.g.has_node(m2_node):
                sim = float(edge["similarity"])
                co_raters = int(edge.get("co_raters", 0))

                # Adiciona em ambas as direções pois similaridade de Pearson é simétrica
                self.g.add_edge(m1_node, m2_node, rel_type="SIMILAR", similarity=sim, co_raters=co_raters)
                self.g.add_edge(m2_node, m1_node, rel_type="SIMILAR", similarity=sim, co_raters=co_raters)
                added += 2
        print(f"[MovieGraph] {added:,} arestas de SIMILAR adicionadas ao grafo.")

    def count_comedy_ratings_above(self, user_id: int, min_rating: float = 4.0) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Responde à busca solicitada:
        'Quantos filmes o usuário X assistiu são comédia em que ele deu nota de 4 para cima.'

        Retorna:
            (quantidade_de_filmes, lista_com_detalhes)
        """
        user_node = f"u_{user_id}"
        if not self.g.has_node(user_node):
            return 0, []

        comedy_high_rated = []

        # Percorrer todas as arestas de saída do usuário (filmes avaliados)
        for _, movie_node, edge_data in self.g.out_edges(user_node, data=True):
            if edge_data.get("rel_type") == "RATED":
                rating = edge_data.get("rating", 0.0)
                if rating >= min_rating:
                    movie_data = self.g.nodes[movie_node]
                    genres = movie_data.get("genres", [])
                    if "Comedy" in genres:
                        comedy_high_rated.append({
                            "movieId": movie_data.get("movieId"),
                            "title": movie_data.get("title"),
                            "rating": rating,
                            "genres": genres,
                            "split": edge_data.get("split")
                        })

        return len(comedy_high_rated), comedy_high_rated

    def get_similar_movies(self, movie_id: int, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retorna os top_k filmes mais similares a partir das arestas :SIMILAR do grafo.
        """
        movie_node = f"m_{movie_id}"
        if not self.g.has_node(movie_node):
            return []

        neighbors = []
        for _, neighbor_node, edge_data in self.g.out_edges(movie_node, data=True):
            if edge_data.get("rel_type") == "SIMILAR":
                sim = edge_data.get("similarity", 0.0)
                target_m_id = self.g.nodes[neighbor_node].get("movieId")
                target_title = self.g.nodes[neighbor_node].get("title", f"Movie {target_m_id}")
                neighbors.append({
                    "movieId": target_m_id,
                    "title": target_title,
                    "similarity": sim,
                    "co_raters": edge_data.get("co_raters", 0)
                })

        neighbors.sort(key=lambda x: x["similarity"], reverse=True)
        return neighbors[:top_k]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Retorna estatísticas detalhadas do grafo.
        """
        user_count = sum(1 for _, d in self.g.nodes(data=True) if d.get("node_type") == "User")
        movie_count = sum(1 for _, d in self.g.nodes(data=True) if d.get("node_type") == "Movie")
        genre_count = sum(1 for _, d in self.g.nodes(data=True) if d.get("node_type") == "Genre")

        rated_count = sum(1 for _, _, d in self.g.edges(data=True) if d.get("rel_type") == "RATED")
        genre_rel_count = sum(1 for _, _, d in self.g.edges(data=True) if d.get("rel_type") == "OF_GENRE")
        similar_count = sum(1 for _, _, d in self.g.edges(data=True) if d.get("rel_type") == "SIMILAR")

        return {
            "total_nodes": self.g.number_of_nodes(),
            "total_edges": self.g.number_of_edges(),
            "users": user_count,
            "movies": movie_count,
            "genres": genre_count,
            "rated_edges": rated_count,
            "of_genre_edges": genre_rel_count,
            "similar_edges": similar_count
        }

    def visualize_subgraph(self, user_id: int, max_movies: int = 6, output_path: str = "graph_sample.png"):
        """
        Gera uma imagem ilustrativa da vizinhança do grafo em torno de um usuário X:
        (User) -[RATED]-> (Movie) -[OF_GENRE]-> (Genre)
                            |
                         [SIMILAR]
                            v
                         (Movie)
        """
        user_node = f"u_{user_id}"
        if not self.g.has_node(user_node):
            print(f"[MovieGraph] Usuário {user_id} não encontrado para visualização.")
            return

        sub_nodes = {user_node}
        # Pegar alguns filmes avaliados
        rated_edges = [
            (u, v, d) for u, v, d in self.g.out_edges(user_node, data=True)
            if d.get("rel_type") == "RATED"
        ][:max_movies]

        for _, m_node, _ in rated_edges:
            sub_nodes.add(m_node)
            # Adicionar gêneros desses filmes
            for _, g_node, gd in self.g.out_edges(m_node, data=True):
                if gd.get("rel_type") == "OF_GENRE":
                    sub_nodes.add(g_node)
            # Adicionar 1 vizinho similar se houver
            for _, s_node, sd in self.g.out_edges(m_node, data=True):
                if sd.get("rel_type") == "SIMILAR":
                    sub_nodes.add(s_node)
                    break

        subgraph = self.g.subgraph(sub_nodes)
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(subgraph, seed=42, k=0.8)

        # Cores por tipo de nó
        color_map = []
        labels = {}
        for node in subgraph.nodes():
            n_data = self.g.nodes[node]
            n_type = n_data.get("node_type")
            if n_type == "User":
                color_map.append("#FF6B6B")
                labels[node] = f"User {n_data.get('userId')}"
            elif n_type == "Movie":
                color_map.append("#4D96FF")
                title = n_data.get("title", "")
                short_title = (title[:18] + "..") if len(title) > 20 else title
                labels[node] = short_title
            elif n_type == "Genre":
                color_map.append("#6BCB77")
                labels[node] = n_data.get("name")
            else:
                color_map.append("#FFD93D")
                labels[node] = node

        nx.draw_networkx_nodes(subgraph, pos, node_color=color_map, node_size=1600, alpha=0.9)
        nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=8, font_weight="bold")

        # Desenhar arestas por tipo
        rated_e = [(u, v) for u, v, d in subgraph.edges(data=True) if d.get("rel_type") == "RATED"]
        genre_e = [(u, v) for u, v, d in subgraph.edges(data=True) if d.get("rel_type") == "OF_GENRE"]
        sim_e = [(u, v) for u, v, d in subgraph.edges(data=True) if d.get("rel_type") == "SIMILAR"]

        nx.draw_networkx_edges(subgraph, pos, edgelist=rated_e, edge_color="#FF6B6B", width=1.8, arrows=True)
        nx.draw_networkx_edges(subgraph, pos, edgelist=genre_e, edge_color="#6BCB77", width=1.2, arrows=True, style="dashed")
        nx.draw_networkx_edges(subgraph, pos, edgelist=sim_e, edge_color="#4D96FF", width=2.2, arrows=True)

        plt.title(f"Amostra do Grafo Heterogêneo centrado no Usuário {user_id}\n(Vermelho=User, Azul=Movie, Verde=Genre)",
                  fontsize=12, fontweight="bold")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[MovieGraph] Visualização da sub-rede salva em: {output_path}")

