#!/usr/bin/env python3
"""
Gera o dataset simulado de pacientes cardíacos do projeto CardioIA (Fase 1 – Parte 1).

Os dados são 100% SINTÉTICOS: nenhum paciente real está representado. As faixas de
valores e as correlações entre as variáveis foram inspiradas em bases públicas clássicas
(UCI Heart Disease / Cleveland e Framingham Heart Study) e nos valores de referência das
diretrizes da Sociedade Brasileira de Cardiologia, para que o dataset seja realista o
suficiente para alimentar os módulos de IA das próximas fases do projeto.

A geração é determinística (semente fixa): rodar o script de novo produz o mesmo arquivo.

Uso:
    python3 scripts/gerar_dataset.py               # 500 linhas em data/dataset_cardiologico.csv
    python3 scripts/gerar_dataset.py --n 1000      # outro tamanho
    python3 scripts/gerar_dataset.py --seed 7      # outra semente
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SAIDA_PADRAO = Path(__file__).resolve().parent.parent / "data" / "dataset_cardiologico.csv"


def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-z))


def gerar_dataset(n=500, seed=42):
    rng = np.random.default_rng(seed)

    # --- Demografia ---------------------------------------------------------
    idade = np.clip(rng.normal(55, 11, n).round(), 29, 80).astype(int)
    sexo = np.where(rng.random(n) < 0.55, "M", "F")
    homem = (sexo == "M").astype(int)

    # --- Fatores de risco de base -------------------------------------------
    historico_familiar_dcv = (rng.random(n) < 0.30).astype(int)
    tabagismo = (rng.random(n) < np.where(homem, 0.30, 0.18)).astype(int)
    imc = np.clip(rng.normal(27, 4.5, n), 17, 45).round(1)
    # Diabetes fica mais provável com idade e IMC altos
    p_diabetes = sigmoide(-1.8 + 0.03 * (idade - 50) + 0.10 * (imc - 27))
    diabetes = (rng.random(n) < p_diabetes).astype(int)
    # Evento cardíaco prévio (infarto, angina, revascularização): idade, família e sexo
    p_evento = sigmoide(-2.6 + 0.05 * (idade - 50) + 0.9 * historico_familiar_dcv + 0.4 * homem)
    historico_doenca_cardiaca = (rng.random(n) < p_evento).astype(int)

    # --- Sinais vitais (monitoramento / IoT) --------------------------------
    pressao_sistolica = np.clip(
        110 + 0.7 * (idade - 29) + 1.2 * (imc - 27) + rng.normal(0, 12, n), 90, 200
    ).round().astype(int)
    pressao_diastolica = np.clip(
        0.55 * pressao_sistolica + rng.normal(8, 7, n), 55, 120
    ).round().astype(int)
    frequencia_cardiaca_repouso = np.clip(
        rng.normal(72, 10, n) + 3 * tabagismo + 0.3 * (imc - 27), 45, 120
    ).round().astype(int)
    # FC máxima no teste de esforço cai com a idade (~220 - idade) e com doença prévia
    frequencia_cardiaca_maxima = np.clip(
        220 - idade + rng.normal(0, 12, n) - 8 * historico_doenca_cardiaca, 90, 202
    ).round().astype(int)

    # --- Exames laboratoriais -----------------------------------------------
    colesterol_total = np.clip(
        175 + 0.8 * (idade - 29) + 1.5 * (imc - 27) + rng.normal(0, 32, n), 120, 400
    ).round().astype(int)
    colesterol_hdl = np.clip(
        rng.normal(np.where(homem, 45, 55), 12) - 6 * tabagismo, 25, 100
    ).round().astype(int)
    colesterol_ldl = np.clip(
        colesterol_total - colesterol_hdl - rng.normal(30, 10, n), 50, 250
    ).round().astype(int)
    glicemia_jejum = np.clip(
        np.where(diabetes, rng.normal(150, 35, n), rng.normal(92, 10, n)), 65, 300
    ).round().astype(int)

    # Diagnóstico de hipertensão pelo corte das diretrizes (>= 140/90 mmHg)
    hipertensao = ((pressao_sistolica >= 140) | (pressao_diastolica >= 90)).astype(int)

    # --- Desfecho: risco cardíaco elevado -----------------------------------
    # Modelo logístico com ruído: quanto mais fatores de risco, maior a chance de 1.
    # É o que dá "sinal" para os modelos de Machine Learning das próximas fases.
    z = (
        -1.7
        + 0.05 * (idade - 50)
        + 0.6 * homem
        + 0.015 * (pressao_sistolica - 120)
        + 0.008 * (colesterol_total - 200)
        - 0.02 * (colesterol_hdl - 50)
        + 0.7 * diabetes
        + 0.8 * tabagismo
        + 0.6 * historico_familiar_dcv
        + 1.2 * historico_doenca_cardiaca
        + 0.04 * (imc - 27)
        + rng.normal(0, 0.8, n)
    )
    risco_cardiaco = (rng.random(n) < sigmoide(z)).astype(int)
    doente = risco_cardiaco == 1

    # --- Sintomas e ECG: dependem do desfecho -------------------------------
    tipos_dor = np.array(["tipica", "atipica", "nao_anginosa", "assintomatico"])
    tipo_dor_peito = np.where(
        doente,
        rng.choice(tipos_dor, n, p=[0.45, 0.25, 0.10, 0.20]),
        rng.choice(tipos_dor, n, p=[0.08, 0.20, 0.27, 0.45]),
    )
    dispneia = (rng.random(n) < np.where(doente, 0.55, 0.15)).astype(int)
    palpitacoes = (rng.random(n) < np.where(doente, 0.35, 0.15)).astype(int)
    angina_esforco = (rng.random(n) < np.where(doente, 0.50, 0.10)).astype(int)
    depressao_st = np.clip(
        np.where(doente, rng.gamma(2.0, 0.8, n), rng.gamma(1.2, 0.4, n)), 0, 6
    ).round(1)
    saturacao_o2 = np.clip(rng.normal(97, 1.2, n) - 1.5 * dispneia, 88, 100).round().astype(int)

    # --- Metadados da coleta ------------------------------------------------
    data_coleta = pd.to_datetime("2026-08-01") + pd.to_timedelta(rng.integers(0, 31, n), unit="D")

    return pd.DataFrame(
        {
            "id_paciente": [f"P{i:04d}" for i in range(1, n + 1)],
            "data_coleta": data_coleta.strftime("%Y-%m-%d"),
            "idade": idade,
            "sexo": sexo,
            "imc": imc,
            "pressao_sistolica": pressao_sistolica,
            "pressao_diastolica": pressao_diastolica,
            "frequencia_cardiaca_repouso": frequencia_cardiaca_repouso,
            "frequencia_cardiaca_maxima": frequencia_cardiaca_maxima,
            "saturacao_o2": saturacao_o2,
            "colesterol_total": colesterol_total,
            "colesterol_hdl": colesterol_hdl,
            "colesterol_ldl": colesterol_ldl,
            "glicemia_jejum": glicemia_jejum,
            "tabagismo": tabagismo,
            "diabetes": diabetes,
            "hipertensao": hipertensao,
            "historico_familiar_dcv": historico_familiar_dcv,
            "historico_doenca_cardiaca": historico_doenca_cardiaca,
            "tipo_dor_peito": tipo_dor_peito,
            "dispneia": dispneia,
            "palpitacoes": palpitacoes,
            "angina_esforco": angina_esforco,
            "depressao_st": depressao_st,
            "risco_cardiaco": risco_cardiaco,
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Gera o dataset simulado de pacientes cardíacos.")
    parser.add_argument("--n", type=int, default=500, help="número de linhas (padrão: 500)")
    parser.add_argument("--seed", type=int, default=42, help="semente aleatória (padrão: 42)")
    parser.add_argument("--saida", type=Path, default=SAIDA_PADRAO, help="caminho do CSV de saída")
    args = parser.parse_args()

    df = gerar_dataset(args.n, args.seed)
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.saida, index=False, encoding="utf-8")

    print(f"Dataset salvo em {args.saida}: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print(f"  risco_cardiaco = 1: {df['risco_cardiaco'].mean():.1%}")
    print(f"  hipertensao    = 1: {df['hipertensao'].mean():.1%}")
    print(f"  diabetes       = 1: {df['diabetes'].mean():.1%}")
    print(f"  tabagismo      = 1: {df['tabagismo'].mean():.1%}")
    print(f"  sexo M / F      : {(df['sexo'] == 'M').mean():.1%} / {(df['sexo'] == 'F').mean():.1%}")


if __name__ == "__main__":
    main()
