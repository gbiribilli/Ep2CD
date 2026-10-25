import sys
import os
import argparse
import numpy as np

from data_loader import load_data, split_data
from graph_model import MovieGraph
from recommender import ItemItemKNNRecommender
from evaluation import Evaluator
from neo4j_manager import Neo4jManager


def main():
    parser = argparse.ArgumentParser(description="MovieLens Recommender System & Graph Modeling")
    parser.add_argument("--test-user", type=int, default=1, help="ID do usuário para testar buscas e recomendações (padrão: 1)")
    parser.add_argument("--k-neighbors", type=int, default=20, help="Quantidade K de vizinhos mais próximos no KNN (padrão: 20)")
    parser.add_argument("--threshold", type=float, default=4.0, help="Limiar de relevância para Precisão e Recall (padrão: 4.0)")
    parser.add_argument("--use-neo4j", action="store_true", help="Se informado, conecta ao Neo4j local e exporta o grafo")
    parser.add_argument("--neo4j-uri", type=str, default="bolt://localhost:7687", help="URI do Neo4j")
    parser.add_argument("--neo4j-user", type=str, default="neo4j", help="Usuário do Neo4j")
    parser.add_argument("--neo4j-password", type=str, default="password", help="Senha do Neo4j")
    args = parser.parse_args()

    print("=" * 80)
    print(" SISTEMA DE RECOMENDAÇÃO EM GRAFO - MOVIELENS")
    print(" Disciplina: Ciência de Dados - EP 2")
    print("=" * 80)

    # ---------------------------------------------------------
    # ETAPA 1: CARREGAMENTO DOS DADOS E SPLIT 80% / 20%
    # ---------------------------------------------------------
    print("\n>>> ETAPA 1: Carregamento do Dataset e Divisão Treino/Teste (80%/20%)")
    movies_df, ratings_df = load_data()
    train_df, test_df = split_data(ratings_df, test_size=0.2, random_state=42)

    # ---------------------------------------------------------
    # ETAPA 2: MODELAGEM DO GRAFO
    # ---------------------------------------------------------
    print("\n>>> ETAPA 2: Construção da Modelagem do Grafo")
    graph = MovieGraph()
   
    graph.build_from_dfs(movies_df, train_df)

    # "Quantos filmes o usuário X assistiu são comédia em que ele deu nota de 4 para cima."
    print("\n" + "-" * 75)
    print(f" CONSULTA EM GRAFO: Filmes de Comédia com Nota >= {args.threshold} para Usuário {args.test_user}")
    print("-" * 75)
    count, comedy_movies = graph.count_comedy_ratings_above(user_id=args.test_user, min_rating=args.threshold)
    print(f"-> O Usuário {args.test_user} avaliou {count} filmes de Comédia com nota >= {args.threshold}:")
    for m in comedy_movies[:10]:
        print(f"   * [{m['movieId']}] {m['title']} | Nota dada: {m['rating']} | Gêneros: {', '.join(m['genres'])}")
    if count > 10:
        print(f"   ... e mais {count - 10} filmes.")

    # ---------------------------------------------------------
    # ETAPA 3: SISTEMA DE RECOMENDAÇÃO (PEARSON + REGRESSÃO KNN)
    # ---------------------------------------------------------
    print("\n>>> ETAPA 3: Ajuste do Recomendador (Pearson + Regressão KNN)")
    rec = ItemItemKNNRecommender(k_neighbors=args.k_neighbors, min_common_users=2)
    rec.fit(train_df, movies_df=movies_df)

    # Extrair arestas de similaridade calculadas e adicionar ao grafo (:SIMILAR)
    sim_edges = rec.extract_similarity_edges(max_edges_per_movie=8, min_similarity=0.25, min_co_raters=3, max_total_edges=15000)
    graph.add_similarity_edges(sim_edges)

    stats = graph.get_statistics()
    print("\n[Resumo do Grafo Heterogêneo]:")
    print(f" - Total de Nós: {stats['total_nodes']:,} (Users: {stats['users']:,}, Movies: {stats['movies']:,}, Genres: {stats['genres']:,})")
    print(f" - Total de Arestas: {stats['total_edges']:,} (RATED: {stats['rated_edges']:,}, OF_GENRE: {stats['of_genre_edges']:,}, SIMILAR: {stats['similar_edges']:,})")

    
    print("\nFilmes similares conectados a 'Toy Story (1995)' no Grafo:")
    sim_to_toy = graph.get_similar_movies(movie_id=1, top_k=5)
    for s in sim_to_toy:
        print(f"   * {s['title']} | Similaridade: {s['similarity']:.4f} | Usuários em Comum: {s['co_raters']}")

    # ---------------------------------------------------------
    # ETAPA 4: AVALIAÇÃO DO MODELO NO CONJUNTO DE TESTE (20%)
    # ---------------------------------------------------------
    print("\n>>> ETAPA 4: Avaliação do Modelo no Conjunto de Teste (20%)")
    test_predictions = rec.predict_batch(test_df, k=args.k_neighbors)
    y_true = test_df["rating"].values

    
    metrics_40 = Evaluator.evaluate_threshold_classification(y_true, test_predictions, threshold=4.0)
    metrics_35 = Evaluator.evaluate_threshold_classification(y_true, test_predictions, threshold=3.5)
    top_k_metrics = Evaluator.evaluate_top_k(test_df, test_predictions, k_values=[5, 10, 20], threshold=4.0)

    print("\n" + "=" * 75)
    print(f" RESULTADOS DA AVALIAÇÃO (Limiar de Relevância >= {args.threshold}):")
    print("=" * 75)
    print(f" - Precisão (Precision):        {metrics_40['precision']*100:.2f}%")
    print(f" - Cobertura (Recall):          {metrics_40['recall']*100:.2f}%")
    print(f" - F1-Score:                    {metrics_40['f1_score']:.4f}")
    print(f" - Erro Médio Absoluto (MAE):   {metrics_40['mae']:.4f}")
    print(f" - Raiz do Erro Quadrático (RMSE): {metrics_40['rmse']:.4f}")
    print(f" - Verdadeiros Positivos (TP):  {metrics_40['tp']:,}")
    print(f" - Falsos Positivos (FP):       {metrics_40['fp']:,}")
    print(f" - Falsos Negativos (FN):       {metrics_40['fn']:,}")
    print(f" - Verdadeiros Negativos (TN):  {metrics_40['tn']:,}")

    print("\n[Métricas de Ranking - Top-K por Usuário no Teste (Relevante se nota >= 4.0)]:")
    for k in [5, 10, 20]:
        p = top_k_metrics[f'precision@{k}'] * 100
        r = top_k_metrics[f'recall@{k}'] * 100
        print(f" - Top-{k:2d}:  Precisão@{k:2d} = {p:.2f}%  |  Recall@{k:2d} = {r:.2f}%")

    print("\n[Comparativo com Limiar >= 3.5]:")
    print(f" - Precisão: {metrics_35['precision']*100:.2f}% | Recall: {metrics_35['recall']*100:.2f}% | F1: {metrics_35['f1_score']:.4f}")

    
    eval_chart = "evaluation_metrics.png"
    subgraph_chart = "graph_sample.png"
    Evaluator.plot_evaluation(metrics_35, metrics_40, top_k_metrics, output_path=eval_chart)
    graph.visualize_subgraph(user_id=args.test_user, max_movies=5, output_path=subgraph_chart)

    # ---------------------------------------------------------
    # ETAPA 5: DEMONSTRAÇÃO DE RECOMENDAÇÃO PERSONALIZADA
    # ---------------------------------------------------------
    print("\n" + "=" * 75)
    print(f" TOP 10 RECOMENDAÇÕES PERSONALIZADAS PARA O USUÁRIO {args.test_user}")
    print("=" * 75)
    recs = rec.recommend(user_id=args.test_user, top_n=10, k=args.k_neighbors)
    for r in recs:
        print(f" {r['rank']:2d}. {r['title']} | Nota Prevista: {r['predicted_rating']} (Média Geral: {r['avg_rating']})")

    # ---------------------------------------------------------
    # ETAPA 6: INTEGRAÇÃO COM NEO4J (OPCIONAL)
    # ---------------------------------------------------------
    if args.use_neo4j:
        print("\n>>> ETAPA 6: Exportando para o Banco de Dados de Grafos Neo4j...")
        neo = Neo4jManager(uri=args.neo4j_uri, user=args.neo4j_user, password=args.neo4j_password)
        if neo.connect():
            try:
                neo.create_schema()
                neo.load_movies(movies_df)
                neo.load_ratings(train_df)
                neo.load_similarity_edges(sim_edges)
                cypher_res = neo.query_user_comedy_high_rated(user_id=args.test_user, min_rating=args.threshold)
                print(f"[Neo4j Cypher] Resultado confirmado no Neo4j: {cypher_res['count']} comédias >= {args.threshold}")
            finally:
                neo.close()

    print("\n" + "=" * 80)
    print(" EXECUÇÃO CONCLUÍDA COM SUCESSO!")
    print(f" Gráficos gerados: '{eval_chart}' e '{subgraph_chart}'")
    print("=" * 80)


if __name__ == "__main__":
    main()

