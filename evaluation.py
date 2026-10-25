"""
evaluation.py
=============
Módulo de Avaliação do Sistema de Recomendação.

Calcula as métricas exigidas:
- Precisão (Precision)
- Cobertura / Revocação (Recall)
- F1-Score
- Precisão@K e Recall@K (Ranking)
- MAE e RMSE (Erro de Regressão)
Gera gráficos ilustrativos dos resultados.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, mean_absolute_error


class Evaluator:
    """
    Avalia as predições geradas pelo recomendador contra o conjunto de teste de 20%.
    """

    @staticmethod
    def evaluate_threshold_classification(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 4.0) -> Dict[str, Any]:
        """
        Avalia o modelo sob a perspectiva de classificação binária:
        - Relevante: nota real >= threshold
        - Recomendado: nota inferida >= threshold
        """
        y_true_bin = (y_true >= threshold).astype(int)
        y_pred_bin = (y_pred >= threshold).astype(int)

        prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        cm = confusion_matrix(y_true_bin, y_pred_bin)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        return {
            "threshold": threshold,
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "mae": float(mae),
            "rmse": float(rmse),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "confusion_matrix": cm
        }

    @staticmethod
    def evaluate_top_k(test_df: pd.DataFrame, predictions: np.ndarray, k_values: List[int] = [5, 10, 20], threshold: float = 4.0) -> Dict[str, float]:
        """
        Calcula Precision@K e Recall@K agregadas por usuário no conjunto de teste.
        """
        eval_df = test_df.copy()
        eval_df["pred"] = predictions

        results = {}
        for k in k_values:
            precisions = []
            recalls = []

            for _, group in eval_df.groupby("userId"):
                # Total de itens relevantes do usuário no teste
                relevant_items = set(group[group["rating"] >= threshold]["movieId"])
                if not relevant_items:
                    continue

                # Itens ordenados pela nota prevista (Top-K)
                top_k_group = group.sort_values(by="pred", ascending=False).head(k)
                recommended_items = set(top_k_group["movieId"])

                hits = len(relevant_items.intersection(recommended_items))

                precisions.append(hits / min(k, len(top_k_group)))
                recalls.append(hits / len(relevant_items))

            avg_p = float(np.mean(precisions)) if precisions else 0.0
            avg_r = float(np.mean(recalls)) if recalls else 0.0

            results[f"precision@{k}"] = avg_p
            results[f"recall@{k}"] = avg_r

        return results

    @staticmethod
    def plot_evaluation(metrics_35: Dict[str, Any], metrics_40: Dict[str, Any], top_k_res: Dict[str, float], output_path: str = "evaluation_metrics.png"):
        """
        Gera um painel gráfico com a matriz de confusão, métricas comparativas e curva Precision@K.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Gráfico 1: Métricas de Classificação por Limiar
        labels = ["Precisão", "Recall", "F1-Score"]
        vals_35 = [metrics_35["precision"], metrics_35["recall"], metrics_35["f1_score"]]
        vals_40 = [metrics_40["precision"], metrics_40["recall"], metrics_40["f1_score"]]

        x = np.arange(len(labels))
        width = 0.35
        axes[0].bar(x - width/2, vals_35, width, label="Limiar = 3.5", color="#4D96FF")
        axes[0].bar(x + width/2, vals_40, width, label="Limiar = 4.0", color="#FF6B6B")
        axes[0].set_ylabel("Pontuação")
        axes[0].set_title("Comparação de Métricas (Thresholds 3.5 e 4.0)", fontweight="bold")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels)
        axes[0].set_ylim(0, 1.05)
        axes[0].legend()
        axes[0].grid(axis="y", linestyle="--", alpha=0.5)

        for rect in axes[0].patches:
            h = rect.get_height()
            if h > 0:
                axes[0].annotate(f"{h:.2f}", (rect.get_x() + rect.get_width() / 2, h),
                                 ha="center", va="bottom", fontsize=9)

        # Gráfico 2: Matriz de Confusão para Threshold 4.0
        cm = metrics_40["confusion_matrix"]
        im = axes[1].imshow(cm, cmap="Blues")
        axes[1].set_title(f"Matriz de Confusão (Limiar >= 4.0)\nTotal de Testes: {cm.sum():,}", fontweight="bold")
        axes[1].set_xticks([0, 1])
        axes[1].set_yticks([0, 1])
        axes[1].set_xticklabels(["Não Rec (<4)", "Rec (>=4)"])
        axes[1].set_yticklabels(["Não Rel (<4)", "Rel (>=4)"])
        for i in range(2):
            for j in range(2):
                axes[1].text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                             color="white" if cm[i, j] > cm.max()/2 else "black", fontweight="bold", fontsize=11)
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        # Gráfico 3: Precision@K e Recall@K
        ks = [5, 10, 20]
        p_at_k = [top_k_res.get(f"precision@{k}", 0) for k in ks]
        r_at_k = [top_k_res.get(f"recall@{k}", 0) for k in ks]

        axes[2].plot(ks, p_at_k, marker="o", linewidth=2.5, color="#6BCB77", label="Precisão@K")
        axes[2].plot(ks, r_at_k, marker="s", linewidth=2.5, color="#FFD93D", label="Recall@K")
        axes[2].set_xlabel("K (Tamanho da Lista de Recomendação)")
        axes[2].set_ylabel("Valor da Métrica")
        axes[2].set_title("Ranking: Precisão@K vs Recall@K (Limiar >= 4.0)", fontweight="bold")
        axes[2].set_xticks(ks)
        axes[2].set_ylim(0, 1.05)
        axes[2].grid(True, linestyle="--", alpha=0.5)
        axes[2].legend()

        for k, p, r in zip(ks, p_at_k, r_at_k):
            axes[2].annotate(f"{p:.2f}", (k, p), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9)
            axes[2].annotate(f"{r:.2f}", (k, r), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[Evaluator] Gráfico comparativo de métricas salvo em: {output_path}")

