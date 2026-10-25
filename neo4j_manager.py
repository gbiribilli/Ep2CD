"""
neo4j_manager.py
================
Gerenciador de integração com banco de dados Neo4j (Local ou Cloud / Aura).
Permite criar o esquema, índices, carregar nós e arestas em lotes (batch UNWIND)
e executar consultas Cypher complexas.
"""

from typing import List, Dict, Any, Optional

try:
    from neo4j import GraphDatabase, Driver
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    Driver = Any


class Neo4jManager:
    """
    Controlador para operações no Neo4j com suporte a carregamento em lote e Cypher.
    """

    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver: Optional[Driver] = None
        self.is_connected = False

    def connect(self) -> bool:
        """
        Tenta estabelecer conexão com o servidor Neo4j.
        """
        if not NEO4J_AVAILABLE:
            print("[Neo4j] Biblioteca 'neo4j' não encontrada. Instale com: pip install neo4j")
            return False

        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            self.is_connected = True
            print(f"[Neo4j] Conexão estabelecida com sucesso em {self.uri}")
            return True
        except Exception as e:
            print(f"[Neo4j] Não foi possível conectar ao Neo4j ({self.uri}): {e}")
            print("[Neo4j] Dica: Certifique-se de que o Neo4j Desktop, container Docker ou instância Aura está ativo.")
            self.is_connected = False
            return False

    def close(self):
        """
        Fecha a conexão com o banco de dados.
        """
        if self.driver:
            self.driver.close()
            self.is_connected = False

    def create_schema(self):
        """
        Cria restrições de unicidade e índices para nós e arestas.
        Garante performance ideal nas buscas e merges.
        """
        if not self.is_connected:
            return

        queries = [
            "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.userId IS UNIQUE",
            "CREATE CONSTRAINT movie_id_unique IF NOT EXISTS FOR (m:Movie) REQUIRE m.movieId IS UNIQUE",
            "CREATE CONSTRAINT genre_name_unique IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE",
            "CREATE INDEX movie_title_idx IF NOT EXISTS FOR (m:Movie) ON (m.title)"
        ]

        with self.driver.session() as session:
            for q in queries:
                try:
                    session.run(q)
                except Exception as e:
                    print(f"[Neo4j] Aviso ao criar constraint: {e}")
        print("[Neo4j] Constraints e índices verificados/criados.")

    def load_movies(self, movies_df, batch_size: int = 1000):
        """
        Insere filmes e seus respectivos nós/arestas de gênero.
        """
        if not self.is_connected:
            return

        query = """
        UNWIND $batch AS row
        MERGE (m:Movie {movieId: row.movieId})
        SET m.title = row.title, m.genres = row.genres
        WITH m, row
        UNWIND row.genres_list AS gName
        MERGE (g:Genre {name: gName})
        MERGE (m)-[:OF_GENRE]->(g)
        """

        records = []
        for _, r in movies_df.iterrows():
            records.append({
                "movieId": int(r["movieId"]),
                "title": str(r["title"]),
                "genres": str(r["genres"]),
                "genres_list": r["genres_list"] if isinstance(r["genres_list"], list) else []
            })

        with self.driver.session() as session:
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                session.run(query, batch=batch)
                print(f"[Neo4j] Carregando filmes... {min(i + batch_size, len(records))}/{len(records)}", end="\r")
        print("\n[Neo4j] Todos os filmes e gêneros foram carregados.")

    def load_ratings(self, ratings_df, batch_size: int = 2500):
        """
        Insere usuários e arestas RATED em lotes otimizados.
        """
        if not self.is_connected:
            return

        query = """
        UNWIND $batch AS row
        MERGE (u:User {userId: row.userId})
        WITH u, row
        MATCH (m:Movie {movieId: row.movieId})
        MERGE (u)-[r:RATED]->(m)
        SET r.rating = toFloat(row.rating),
            r.timestamp = toInteger(row.timestamp),
            r.split = row.split
        """

        records = []
        for _, r in ratings_df.iterrows():
            records.append({
                "userId": int(r["userId"]),
                "movieId": int(r["movieId"]),
                "rating": float(r["rating"]),
                "timestamp": int(r["timestamp"]),
                "split": str(r.get("split", "train"))
            })

        with self.driver.session() as session:
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                session.run(query, batch=batch)
                print(f"[Neo4j] Carregando avaliações... {min(i + batch_size, len(records))}/{len(records)}", end="\r")
        print("\n[Neo4j] Avaliações carregadas.")

    def load_similarity_edges(self, similarity_edges: List[Dict[str, Any]], batch_size: int = 2500):
        """
        Insere arestas de similaridade (:Movie)-[:SIMILAR]->(:Movie).
        """
        if not self.is_connected:
            return

        query = """
        UNWIND $batch AS row
        MATCH (m1:Movie {movieId: row.movie1})
        MATCH (m2:Movie {movieId: row.movie2})
        MERGE (m1)-[s:SIMILAR]->(m2)
        SET s.similarity = toFloat(row.similarity),
            s.co_raters = toInteger(row.co_raters)
        """

        with self.driver.session() as session:
            for i in range(0, len(similarity_edges), batch_size):
                batch = similarity_edges[i:i + batch_size]
                session.run(query, batch=batch)
                print(f"[Neo4j] Carregando similaridades... {min(i + batch_size, len(similarity_edges))}/{len(similarity_edges)}", end="\r")
        print("\n[Neo4j] Arestas de similaridade carregadas com sucesso.")

    def query_user_comedy_high_rated(self, user_id: int, min_rating: float = 4.0) -> Dict[str, Any]:
        """
        Executa a busca solicitada em Cypher:
        'Quantos filmes o usuário X assistiu são comédia em que ele deu nota de 4 para cima.'
        """
        if not self.is_connected:
            return {"count": 0, "movies": [], "error": "Not connected"}

        query = """
        MATCH (u:User {userId: $userId})-[r:RATED]->(m:Movie)-[:OF_GENRE]->(g:Genre {name: 'Comedy'})
        WHERE r.rating >= $minRating
        RETURN count(DISTINCT m) AS total_comedy,
               collect(DISTINCT {movieId: m.movieId, title: m.title, rating: r.rating}) AS movies
        """

        with self.driver.session() as session:
            result = session.run(query, userId=user_id, minRating=min_rating)
            record = result.single()
            if record:
                return {
                    "count": record["total_comedy"],
                    "movies": record["movies"]
                }
            return {"count": 0, "movies": []}

    def query_top_similar(self, movie_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Busca os filmes mais similares ao movieId informado via arestas :SIMILAR.
        """
        if not self.is_connected:
            return []

        query = """
        MATCH (m1:Movie {movieId: $movieId})-[s:SIMILAR]->(m2:Movie)
        RETURN m2.movieId AS movieId, m2.title AS title, s.similarity AS similarity, s.co_raters AS co_raters
        ORDER BY s.similarity DESC
        LIMIT $limit
        """

        with self.driver.session() as session:
            result = session.run(query, movieId=movie_id, limit=limit)
            return [dict(r) for r in result]

