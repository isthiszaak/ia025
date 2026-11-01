# Validação dos dados — Member 1

Conferência realizada em 31/10/2026 nos arquivos reais do
[BR-TaxQA-R, revisão `9f0f4263928c6152a40f888da3e63375a24bb0f3`](https://huggingface.co/datasets/unicamp-dl/BR-TaxQA-R/tree/9f0f4263928c6152a40f888da3e63375a24bb0f3).
Os números abaixo vêm da preparação local dos arquivos públicos, sem embeddings.

| Parte | Registros de entrada | Documentos únicos / perguntas aceitas | Descartes |
| --- | ---: | ---: | ---: |
| Normas legais v1.1 | 478 | 461 documentos | 17 textos normalizados idênticos |
| Perguntas RFB 2024 | 715 | 715 perguntas | 0 |
| CARF 2023 | 7.204 | 7.204 documentos | 0 |
| Normas + CARF | 7.682 | 7.665 documentos | 17 |

Não há duplicatas normalizadas entre CARF e normas nesta revisão. A diferença
478 − 461 é explicada integralmente por 17 registros duplicados, sem registros
inválidos ou vazios. A normalização aplica NFC, finais de linha LF e remoção
de espaço nas extremidades; mantém parágrafos e conteúdo interno. O ID de
documento continua sendo `doc_` + SHA256 do texto normalizado.

Todos os nomes dos 478 registros legais e dos 7.204 registros CARF foram
verificados contra `source_aliases` e `source_records` nos documentos preparados.
Para cada origem, seu `filedata` normalizado coincide com o texto canônico.
`source_records` preserva arquivo do dataset, subset, posição e filename;
`manifest.json` lista cada descarte com o ID canônico e sua origem. O filename
alternativo permanece utilizável para resolver referências após deduplicação.

## Fonte e estrutura do CARF

A fonte dos 7.204 casos do plano é
[`acordaos_CARF_2023.json`](https://huggingface.co/datasets/unicamp-dl/BR-TaxQA-R/blob/9f0f4263928c6152a40f888da3e63375a24bb0f3/acordaos_CARF_2023.json),
na mesma revisão do BR-TaxQA-R. Os bytes baixados conferem com o SHA256 publicado:
`2fca2a7f0bc89c08f1eefb04ee4167a98c9fc170af25c1ac2468098c9eb079bd`.

O arquivo é um array JSON; todos os 7.204 objetos têm exatamente `filename` e
`filedata`, ambos strings. Os filenames têm a forma numérica
`10070000217200724_6764220.txt`; não se repetem. `filedata` contém o texto
extraído do acórdão, inclusive ruído de extração que não é corrigido por inferência.
Não há perguntas ou respostas de referência no arquivo CARF.

O [dataset card](https://huggingface.co/datasets/unicamp-dl/BR-TaxQA-R/blob/9f0f4263928c6152a40f888da3e63375a24bb0f3/README.md)
e o [artigo dos autores, seção 2.3](https://arxiv.org/html/2505.15916v1#S2.SS3)
descrevem a coleta de decisões de 2023 a partir do repositório oficial do CARF,
com seleção temática relacionada às perguntas RFB. Essa relação temática não
constitui anotação de relevância por pergunta. A ingestão mantém o identificador
original do arquivo em `case_file_ids`, o ano em `case_year`, `document_type`,
subset, revisão, aliases e URL do arquivo do dataset. O repositório oficial de
origem é registrado, sem construir URLs individuais ou adivinhar números de
acórdão a partir do filename.

## Perguntas e referências

As 715 perguntas, suas respostas e todos os campos originais ficam em
`questions.jsonl`, separados de `corpus.jsonl`. Nem `answer` nem `answer_cleaned`
participam do corpus ou do chunking.

A auditoria de `all_formatted_references` encontrou 598 perguntas com referências
explícitas a arquivos e 117 sem elas. Há 459 filenames distintos referenciados;
todos resolvem para um alias preservado no corpus legal, sem referência não
resolvida. Nenhuma pergunta aponta explicitamente a um filename do CARF.
O manifesto registra essa auditoria. Não foram criados rótulos de relevância
CARF, métricas de qualidade de respostas ou julgamentos baseados em distância.

## Reprodução

Na raiz do repositório, após instalar as dependências:

```powershell
# Corpus legal padrão e perguntas separadas.
& '.\.venv-retrieval\Scripts\python.exe' -m app.retrieval prepare

# Corpus combinado em diretório próprio, reutilizando os arquivos verificados.
& '.\.venv-retrieval\Scripts\python.exe' -m app.retrieval prepare --include-carf --data-dir data/retrieval-carf --raw-dir data/retrieval/raw/9f0f4263928c6152a40f888da3e63375a24bb0f3
```

Se `--raw-dir` for omitido, os arquivos necessários são baixados para o cache
do diretório escolhido. `prepare_dataset(..., include_carf=True)` oferece a
mesma opção pela API Python. O padrão continua sendo apenas normas; a seleção
pequena existente continua disponível para verificações rápidas.

Os manifestos reais estão em `data/retrieval/prepared/manifest.json` e
`data/retrieval-carf/prepared/manifest.json`, ignorados pelo Git junto aos dados.
Os seis testes em `tests/test_retrieval_carf.py` usam registros sintéticos para
verificar opt-in, deduplicação entre subsets, preservação de origens, separação
de gabaritos, referências não resolvidas, repetição e integridade do cache.
A validação dos arquivos públicos e contagens acima foi uma execução real
separada desses testes; ela não mede qualidade de recuperação nem indexação.
