# CardioIA — Fase 1: Batimentos de Dados

Repositório do projeto CardioIA (FIAP — Inteligência Artificial), uma plataforma acadêmica que simula o ecossistema de uma cardiologia moderna: triagem, diagnóstico, monitoramento e previsões médicas apoiadas por Machine Learning, NLP, Visão Computacional e IoT.

Nesta primeira fase o papel é o de cientista de dados hospitalar: levantar, organizar e entender os dados cardiológicos que vão alimentar os módulos inteligentes das fases seguintes, com atenção à governança de dados e ao viés.

**Grupo 63 — Victor Araujo Ferreira da Silva**

## Estrutura do repositório

```
.
├── assets/
│   ├── imagens_amostra/            # Parte 3 – amostra de imagens de ECG (16 arquivos)
│   │   ├── normal/
│   │   ├── infarto_miocardio/
│   │   ├── arritmia/
│   │   └── historico_infarto/
│   └── manifest_imagens.csv        # rastreabilidade da seleção das imagens
├── data/
│   └── dataset_cardiologico.csv   # Parte 1 – dataset numérico (500 linhas x 25 colunas)
├── scripts/
│   └── gerar_dataset.py           # gera o dataset de forma determinística (semente fixa)
├── docs/                          # Parte 2 – textos (.txt) para NLP
├── notebooks/                     # notebooks (Colab/Jupyter) que vão consumir os dados
├── requirements.txt
└── README.md
```

## Parte 1 – Dados Numéricos (IoT)

### Link público para o dataset

Arquivo no repositório: `data/dataset_cardiologico.csv`
Download direto (raw): https://raw.githubusercontent.com/VictorAraujopy/Inteligencia-Cardiologica/main/data/dataset_cardiologico.csv
Formato `.csv` (UTF-8, separador vírgula), 500 linhas e 25 colunas, sem valores ausentes.

### Origem dos dados

Os dados são simulados (sintéticos). Nenhum paciente real está representado.

Dados clínicos reais são protegidos por sigilo médico e pela LGPD (Lei 13.709/2018), e as bases públicas existentes ou são pequenas e antigas (UCI Heart Disease, 1988) ou exigem processo de autorização. Gerar um dataset sintético com estrutura realista resolve o problema de acesso sem expor dados pessoais, o que já é uma decisão de governança de dados.

O dataset foi gerado pelo script `scripts/gerar_dataset.py`, que:

- usa semente aleatória fixa (`seed=42`): rodar o script de novo reproduz exatamente o mesmo arquivo, o que garante rastreabilidade;
- sorteia cada variável dentro de faixas clínicas plausíveis, com distribuições e correlações inspiradas nas bases públicas UCI Heart Disease (Cleveland) e Framingham Heart Study e nos valores de referência das diretrizes da Sociedade Brasileira de Cardiologia (hipertensão a partir de 140/90 mmHg, diabetes a partir de glicemia de jejum de 126 mg/dL, etc.);
- mantém as relações causais esperadas: pressão e colesterol sobem com a idade e o IMC, a frequência cardíaca máxima cai com a idade, o HDL é menor em homens e fumantes, e o desfecho `risco_cardiaco` é sorteado por um modelo logístico que soma os fatores de risco — assim os modelos de Machine Learning das próximas fases têm sinal real para aprender;
- faz os sintomas dependerem do desfecho (dor típica, angina de esforço, dispneia e depressão do segmento ST são mais frequentes nos pacientes de risco), como acontece na clínica.

Resumo do que saiu:

| Indicador | Valor |
|---|---|
| Linhas / colunas | 500 / 25 |
| Idade | 29 a 80 anos (média 54,8) |
| Sexo | 55,6% masculino, 44,4% feminino |
| risco_cardiaco = 1 | 47,8% (classes balanceadas, próximo dos 46% da base de Cleveland) |
| Hipertensão / diabetes / tabagismo | 27,8% / 16,8% / 24,0% |
| Histórico familiar / evento cardíaco prévio | 28,8% / 16,8% |

Para reproduzir ou gerar um dataset maior:

```bash
pip install -r requirements.txt
python3 scripts/gerar_dataset.py            # 500 linhas (padrão)
python3 scripts/gerar_dataset.py --n 1000   # outro tamanho
```

### Dicionário de dados

| Coluna | Tipo | Descrição |
|---|---|---|
| id_paciente | texto | Identificador sintético do paciente (P0001…) |
| data_coleta | data | Data da coleta dos sinais (agosto/2026) |
| idade | inteiro | Idade em anos (29–80) |
| sexo | categórica | M ou F |
| imc | decimal | Índice de massa corporal (kg/m²) |
| pressao_sistolica | inteiro | Pressão arterial sistólica (mmHg) |
| pressao_diastolica | inteiro | Pressão arterial diastólica (mmHg) |
| frequencia_cardiaca_repouso | inteiro | Frequência cardíaca em repouso (bpm) — leitura típica de wearable |
| frequencia_cardiaca_maxima | inteiro | Frequência cardíaca máxima atingida em teste de esforço (bpm) |
| saturacao_o2 | inteiro | Saturação de oxigênio (%) — leitura de oxímetro |
| colesterol_total | inteiro | Colesterol total (mg/dL) |
| colesterol_hdl | inteiro | Colesterol HDL, o "bom" (mg/dL) |
| colesterol_ldl | inteiro | Colesterol LDL, o "ruim" (mg/dL) |
| glicemia_jejum | inteiro | Glicemia de jejum (mg/dL) |
| tabagismo | binária | 1 = fumante |
| diabetes | binária | 1 = diagnóstico de diabetes |
| hipertensao | binária | 1 = pressão ≥ 140/90 mmHg |
| historico_familiar_dcv | binária | 1 = parente de primeiro grau com doença cardiovascular |
| historico_doenca_cardiaca | binária | 1 = evento cardíaco prévio (infarto, angina, revascularização) |
| tipo_dor_peito | categórica | tipica, atipica, nao_anginosa ou assintomatico |
| dispneia | binária | 1 = falta de ar |
| palpitacoes | binária | 1 = palpitações |
| angina_esforco | binária | 1 = dor no peito induzida por esforço |
| depressao_st | decimal | Depressão do segmento ST no ECG de esforço (mm) |
| risco_cardiaco | binária | Variável alvo: 1 = risco cardíaco elevado |

### Variáveis mais relevantes do ponto de vista clínico

As variáveis abaixo são as que mais pesam na prática cardiológica e, por isso, as que um modelo de IA para triagem e previsão de risco precisa enxergar:

- **Idade e sexo** — são os fatores de risco não modificáveis mais fortes. O risco cardiovascular cresce com a idade e é maior em homens até a faixa dos 60 anos. Todo escore clínico (Framingham, SCORE) começa por eles, e um modelo que não os recebe perde a base do risco. Também são as variáveis onde o viés aparece primeiro: um dataset desbalanceado por sexo produz um modelo que erra mais em mulheres, que já têm sintomas menos "clássicos".
- **Pressão arterial (sistólica e diastólica)** — a hipertensão é o principal fator de risco modificável para infarto, AVC e insuficiência cardíaca. É a variável mais barata de monitorar continuamente (IoT), o que a torna a candidata natural para alertas em tempo real.
- **Colesterol (LDL, HDL e total)** — o LDL alto forma a placa aterosclerótica que obstrui as coronárias; o HDL alto protege. A relação entre eles separa melhor os grupos de risco do que o colesterol total sozinho, por isso o dataset traz as três frações.
- **Diabetes e glicemia de jejum** — o diabetes praticamente dobra o risco cardiovascular e costuma mascarar sintomas (isquemia silenciosa). Para a IA é um sinal forte de que o paciente pode estar doente mesmo sem dor no peito.
- **Tabagismo** — fator de risco modificável de grande impacto e de fácil coleta. Além de prever risco, é uma variável de intervenção: o sistema pode priorizar quem mais ganha ao parar de fumar.
- **Histórico de doença cardíaca e histórico familiar** — quem já teve um evento tem risco muito maior de repetir; o histórico familiar captura a carga genética. São as variáveis com maior peso individual no modelo que gerou o dataset e tendem a ser as mais importantes em qualquer classificador treinado sobre ele.
- **Sintomas e ECG de esforço** (tipo de dor no peito, angina de esforço, dispneia, depressão ST) — são o elo entre o dado e a triagem. A dor típica e a depressão do segmento ST são os achados com maior valor preditivo para doença coronariana, e um modelo de NLP/triagem vai extrair esses mesmos sinais de textos e prontuários nas próximas fases.
- **Frequência cardíaca e saturação de O₂** — são os sinais que um dispositivo IoT (smartwatch, oxímetro) entrega de forma contínua. Sozinhos dizem pouco, mas em série temporal permitem detectar arritmias, descompensação e piora antes da consulta, que é o objetivo do módulo de monitoramento remoto do CardioIA.

### Governança de dados e viés

- **Privacidade:** por serem sintéticos, os dados não têm titular e podem ser publicados sem consentimento ou anonimização. Se em fases futuras forem substituídos por dados reais, será preciso base legal (LGPD), anonimização e controle de acesso.
- **Rastreabilidade:** a geração é reprodutível (script versionado + semente fixa), então qualquer número do projeto pode ser auditado de ponta a ponta.
- **Viés conhecido:** as distribuições e as correlações foram escolhidas pelo autor com base em literatura, não medidas em uma população real. O dataset não deve ser usado para tirar conclusões clínicas; serve para construir e testar o pipeline de IA. Também não inclui variáveis como raça/etnia, renda e região, que influenciam o risco cardiovascular real e devem ser avaliadas com cuidado quando dados reais entrarem no projeto.
- **Limitações:** sem valores ausentes, sem ruído de medição e sem erros de digitação — coisas que um dado hospitalar real sempre tem e que precisam ser tratadas na fase de limpeza.

## Parte 2 – Dados Textuais (NLP)

Em construção. Os textos (.txt) ficarão na pasta `docs/`.

## Parte 3 – Dados Visuais (VC)

### As imagens

**120 eletrocardiogramas (ECG)** — o registro gráfico da atividade elétrica do coração. Cada arquivo é um laudo completo com as 12 derivações (I, II, III, aVR, aVL, aVF, V1–V6) na mesma folha, distribuídas em 4 classes: 30 normais, 30 de infarto do miocárdio, 30 de arritmia e 30 com histórico de infarto.

Escolhi ECG porque as variáveis `depressao_st`, `angina_esforco` e `tipo_dor_peito` do dataset numérico da Parte 1 são achados extraídos exatamente desse tipo de exame.

As imagens foram deduplicadas por hash antes da seleção; a classe de infarto tem apenas 30 exames únicos, o que definiu o teto por classe. A origem de cada imagem está registrada em `assets/manifest_imagens.csv`.

**Link do conjunto completo:** https://drive.google.com/drive/folders/1ooh1hQt-Wcepv96eoTZzMFSZhsmqs-8c
**Amostra no repositório:** `assets/imagens_amostra/` (16 imagens, 4 por classe)

### Fonte

> Khan, A. H., & Hussain, M. (2021). *ECG Images dataset of Cardiac Patients* (V2) [Data set]. Mendeley Data. https://doi.org/10.17632/gwbz3fsgp8.2 — licença CC BY 4.0

Imagens capturadas com equipamento EDAN SERIES-3 em unidades de cardiologia no Paquistão, anonimizadas e com rótulos revisados manualmente por profissionais de saúde.

### Justificativa para uso em Visão Computacional

- **Detecção de padrões:** CNNs podem aprender o formato característico de cada onda (P, QRS, T) e reconhecer desvios associados a infarto ou arritmia. Como cada imagem traz as 12 derivações, o modelo aprende também *onde* a alteração aparece, o que ajuda a localizar a região do coração afetada.
- **Identificação de bordas:** técnicas como Canny ou Sobel isolam o traçado do sinal em relação ao fundo (grade milimetrada, cabeçalho do laudo), permitindo extrair a forma de onda e reconstruir o sinal para análise numérica.
- **Reconhecimento de anomalias:** desvios de amplitude, intervalo ou morfologia em relação ao padrão normal podem ser sinalizados automaticamente, funcionando como triagem inicial antes da revisão de um profissional.

Isso é relevante para IA em saúde porque grande parte dos ECGs no mundo real ainda circula como papel ou foto de papel, não como sinal digital. Um sistema capaz de ler a imagem permite escalar a triagem cardiológica e apoiar profissionais em contextos com poucos especialistas — sempre complementando, nunca substituindo, a avaliação clínica humana.

### Governança de dados e viés

- **Origem:** dataset de um único país e um único modelo de aparelho, o que limita a generalização para outras populações e equipamentos.
- **Escassez de classe:** só 30 exames únicos de infarto, pouco para treinar um detector confiável justamente da condição mais grave — nas próximas fases isso pede data augmentation ou uma fonte complementar.
- **Privacidade e licença:** imagens anonimizadas, sem nome de paciente, e licença CC BY 4.0 respeitada com a citação acima.

## Referências

- Janosi, A., Steinbrunn, W., Pfisterer, M., Detrano, R. Heart Disease Data Set (Cleveland). UCI Machine Learning Repository, 1988. https://archive.ics.uci.edu/dataset/45/heart+disease
- Framingham Heart Study. https://www.framinghamheartstudy.org
- Sociedade Brasileira de Cardiologia. Diretrizes Brasileiras de Hipertensão Arterial – 2020.
- Sociedade Brasileira de Cardiologia. Atualização da Diretriz Brasileira de Dislipidemias e Prevenção da Aterosclerose – 2017.
- Brasil. Lei nº 13.709/2018 — Lei Geral de Proteção de Dados Pessoais (LGPD).
- Khan, A. H., & Hussain, M. (2021). *ECG Images dataset of Cardiac Patients* (V2) [Data set]. Mendeley Data. https://doi.org/10.17632/gwbz3fsgp8.2 (CC BY 4.0)
