"""Gera explicações SHAP reproduzíveis para o modelo CatBoost treinado."""
# Para executar: Da raiz do projeto, rode:
# python -m prf_accidents_feature_importance.modeling.explain_shap
# Os resultados serão criados em data/models/catboost/shap/
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

from prf_accidents_feature_importance.config import (
    DIRETORIO_DADOS_PROCESSADOS,
    DIRETORIO_MODELOS,
)
from prf_accidents_feature_importance.features.build_features import (
    COLUNA_ALVO,
    COLUNAS_CATEGORICAS,
    COLUNAS_FEATURES,
)

SEMENTE = 42
ARQUIVO_ENTRADA = DIRETORIO_DADOS_PROCESSADOS / "acidentes_features.parquet"
DIRETORIO_MODELO = DIRETORIO_MODELOS / "catboost"
ARQUIVO_MODELO = DIRETORIO_MODELO / "modelo.cbm"
DIRETORIO_SAIDA = DIRETORIO_MODELO / "shap"


def carregar_teste(caminho: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Reproduz exatamente a divisão 80/10/10 usada no treinamento."""
    dados = pd.read_parquet(caminho).dropna(subset=[COLUNA_ALVO]).copy()
    features = dados[COLUNAS_FEATURES].copy()
    for coluna in COLUNAS_CATEGORICAS:
        features[coluna] = features[coluna].fillna("desconhecido").astype(str)
    alvo = dados[COLUNA_ALVO].astype(str)

    _, x_temporario, _, y_temporario = train_test_split(
        features,
        alvo,
        test_size=0.20,
        random_state=SEMENTE,
        stratify=alvo,
    )
    _, x_teste, _, y_teste = train_test_split(
        x_temporario,
        y_temporario,
        test_size=0.50,
        random_state=SEMENTE,
        stratify=y_temporario,
    )
    colunas_numericas = ["km", "icm", "volume_pedagio", "fator_humano"]
    print(x_teste[colunas_numericas].dtypes)
    print(x_teste[colunas_numericas].isna().sum())
    print(x_teste[colunas_numericas].isna().mean())
    return x_teste, y_teste


def calcular_shap(
    caminho_entrada: Path = ARQUIVO_ENTRADA,
    caminho_modelo: Path = ARQUIVO_MODELO,
    diretorio_saida: Path = DIRETORIO_SAIDA,
) -> None:
    """Calcula SHAP multiclasses no mesmo conjunto de teste do treinamento."""
    diretorio_saida.mkdir(parents=True, exist_ok=True)

    x_teste, y_teste = carregar_teste(caminho_entrada)
    modelo = CatBoostClassifier()
    modelo.load_model(caminho_modelo)

    # Pool preserva a informação das variáveis categóricas para o CatBoost/SHAP.
    pool_teste = Pool(x_teste, y_teste, cat_features=COLUNAS_CATEGORICAS)
    shap_values = modelo.get_feature_importance(pool_teste, type="ShapValues")

    # Para multiclass, o CatBoost retorna uma matriz com uma dimensão por classe
    # e uma coluna adicional com o valor esperado (bias).
    if shap_values.ndim != 3:
        raise ValueError(
            f"Formato SHAP inesperado para multiclass: {shap_values.shape}"
        )

    valores = shap_values[:, :, :-1].transpose(0, 2, 1)
    classes = modelo.classes_

    # Importância global média absoluta, agregada por classe.
    importancia_por_classe = pd.DataFrame(
        {
            "feature": x_teste.columns,
            **{
                str(classe): abs(valores[:, :, indice]).mean(axis=0)
                for indice, classe in enumerate(classes)
            },
        }
    )
    importancia_por_classe["mean_abs_shap"] = importancia_por_classe[
        [str(classe) for classe in classes]
    ].mean(axis=1)
    importancia_por_classe = importancia_por_classe.sort_values(
        "mean_abs_shap", ascending=False
    )
    importancia_por_classe.to_csv(
        diretorio_saida / "shap_feature_importance.csv", index=False
    )

    # Valores por observação para permitir análises posteriores de direção e interação.
    registros = []
    for indice_classe, classe in enumerate(classes):
        quadro = pd.DataFrame(
            valores[:, :, indice_classe],
            columns=x_teste.columns,
        )
        quadro["target_class"] = str(classe)
        quadro["target"] = y_teste.to_numpy()
        quadro["row_id"] = x_teste.index.to_numpy()
        registros.append(quadro)
    pd.concat(registros, ignore_index=True).to_parquet(
        diretorio_saida / "shap_values.parquet", index=False
    )

    mask_plot = (
        x_teste["icm"].notna()
        & x_teste["volume_pedagio"].notna()
    )
    x_plot = x_teste.loc[mask_plot].copy()
    valores_plot = valores[mask_plot.to_numpy()]
    if x_plot.empty:
        raise ValueError(
            "Nenhuma observação disponível para os plots SHAP após o filtro."
        )
    nota_plot = (
        f"Visualização baseada em N = {len(x_plot):,} observações. "
        "Excluídas observações com icm ou volume_pedagio ausentes "
        "somente para os plots."
    )

    # Summary plot global agregado por classe.
    shap.summary_plot(
        np.abs(valores_plot).mean(axis=2),
        x_plot,
        show=False,
    )
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.figtext(0.5, 0.01, nota_plot, ha="center", fontsize=8)
    plt.savefig(diretorio_saida / "shap_summary_global.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Um summary plot para cada classe, permitindo discutir direção dos efeitos.
    for indice_classe, classe in enumerate(classes):
        shap.summary_plot(
            valores_plot[:, :, indice_classe],
            x_plot,
            show=False,
        )
        plt.title(f"SHAP Summary — {classe}")
        plt.tight_layout(rect=(0, 0.08, 1, 1))
        plt.figtext(0.5, 0.01, nota_plot, ha="center", fontsize=8)
        nome = str(classe).replace(" ", "_")
        plt.savefig(
            diretorio_saida / f"shap_summary_{nome}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

    pd.DataFrame({"class": classes}).to_csv(
        diretorio_saida / "classes.csv", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ARQUIVO_ENTRADA)
    parser.add_argument("--model", type=Path, default=ARQUIVO_MODELO)
    parser.add_argument("--output", type=Path, default=DIRETORIO_SAIDA)
    args = parser.parse_args()
    calcular_shap(args.input, args.model, args.output)
