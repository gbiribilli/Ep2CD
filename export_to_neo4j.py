
import argparse
from data_loader import load_data, split_data
from recommender import ItemItemKNNRecommender
from neo4j_manager import Neo4jManager


def main():
    parser = argparse.ArgumentParser(description="Exporta MovieLens e Grafo de Similaridade para o Neo4j.")
    parser.add_argument("--uri", type=str, default="bolt://localhost:7687", help="URI do Neo4j (ex: bolt://localhost:7687 ou neo4j+s://...)")
    parser.add_argument("--user", type=str, default="neo4j", help="Usuário do Neo4j (padrão: neo4j)")
    parser.add_argument("--password", type=str, default="password", help="Senha do Neo4j")
    parser.add_argument("--test-user", type=int, default=1, help="ID do usuário para testar a busca")
    parser.add_argument("--max-sim-edges", type=int, default=15000, help="Quantidade máxima de arestas :SIMILAR a exportar")
    args = parser.parse_args()

    print("=" * 70)
    print(" EXPORTADOR MOVIELENS -> NEO4J")
    print("=" * 70)

    # 1. Carregar dados
    movies_df, ratings_df = load_data()
    train_df, test_df = split_data(ratings_df, test_size=0.2, random_state=42)

    # 2. Treinar recomendador para extrair similaridades entre filmes
    rec = ItemItemKNNRecommender(k_neighbors=20)
    rec.fit(train_df, movies_df=movies_df)
    sim_edges = rec.extract_similarity_edges(max_edges_per_movie=8, min_similarity=0.3, max_total_edges=args.max_sim_edges)

    # 3. Conectar ao Neo4j
    neo = Neo4jManager(uri=args.uri, user=args.user, password=args.password)
    connected = neo.connect()

    if not connected:
        print("\n[ERRO] Não foi possível conectar ao Neo4j.")
        print("Para rodar localmente com Docker:")
        print("  docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest")
        print("Ou crie uma instância gratuita no Neo4j Aura: https://neo4j.com/cloud/aura/")
        return

    try:
        # 4. Criar constraints e esquemas
        neo.create_schema()

        # 5. Carregar Filmes e Gêneros
        neo.load_movies(movies_df)

        # 6. Carregar Usuários e Avaliações
        neo.load_ratings(train_df)

        # 7. Carregar Arestas de Similaridade de Pearson (:SIMILAR)
        neo.load_similarity_edges(sim_edges)

        # 8. Testar Consulta Solicitada no Enunciado
        print("\n" + "=" * 70)
        print(f" TESTE DE BUSCA CYPHER NO NEO4J PARA O USUÁRIO {args.test_user}")
        print("=" * 70)
        res = neo.query_user_comedy_high_rated(user_id=args.test_user, min_rating=4.0)
        print(f"Total de filmes de Comédia com nota >= 4.0 avaliados pelo usuário {args.test_user}: {res['count']}")
        print("Amostra dos filmes:")
        for m in res["movies"][:10]:
            print(f"  - [{m['movieId']}] {m['title']} | Nota: {m['rating']}")

        # 9. Testar recomendação via grafo
        print("\n" + "-" * 70)
        print(" Filmes mais similares a Toy Story (1995) no Grafo Neo4j:")
        print("-" * 70)
        similar_toys = neo.query_top_similar(movie_id=1, limit=5)
        for s in similar_toys:
            print(f"  - {s['title']} | Correlação: {s['similarity']:.4f} | Usuários em Comum: {s['co_raters']}")

    finally:
        neo.close()
        print("\n[Neo4j] Conexão encerrada com sucesso.")


if __name__ == "__main__":
    main()

