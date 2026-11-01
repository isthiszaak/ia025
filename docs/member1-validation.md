# Entrega local — Member 1, Data & Retrieval

Verificação em 31/10/2026 na branch `feature/data-retrieval`, baseada no commit
`3b88b7a`. O checkout inicial estava na `main`, sem alterações locais; a branch
foi aberta a partir de `origin/feature/data-retrieval`. Não foram criados commit,
push ou mudanças nas frentes de geração, avaliação de respostas e interface.

## Alterações e contrato de entrega

- Corpus legal completo validado; 17 duplicatas explicam 478 registros para
  461 textos únicos. Aliases e todas as origens permanecem vinculados ao texto.
- CARF opt-in, após conferir o arquivo publicado: 7.204 documentos e nenhum
  par de pergunta/resposta. Corpus combinado: 7.665 textos únicos. As 715
  perguntas e gabaritos permanecem em arquivo separado, fora do índice.
- Chunking real em tokens 256/512/1024, com tokenizer EmbeddingGemma fixado por
  revisão, hash e versão da biblioteca. Overlap inicial configurável de 50.
  Conteúdo e offsets exatos preservados; entradas completas acima de 2048
  tokens falham explicitamente, com `truncate:false` também no Ollama.
- Índices Chroma persistentes distinguem unidade, tokenizer e parâmetros,
  preservando identidade antiga em caracteres. Mudanças só em top-k reutilizam
  o índice. Reabertura, retomada e rejeição de índices incompletos continuam.
- API `Retriever.retrieve()` preservada: `chunk_id`, `document_id`, `text`,
  `source`, `title`, `metadata`, `distance`, `distance_metric`. Metadados
  incluem origens, aliases, revisão e offsets. CLI salva pergunta, configuração,
  fontes/trechos, tempo e falhas; JSON de console suporta caracteres fora do
  Windows-1252. JSONL salvo mantém UTF-8.
- Comparações variam um fator por vez: `(256,3)`, `(512,3)`, `(1024,3)`,
  `(512,1)`, `(512,5)`. Não foi acrescentada grade experimental completa,
  integração com LLM, avaliador, métrica inventada ou ingestão de segundo dataset.

## Verificações realizadas

Ambiente local: Python 3.12.14, ChromaDB 1.5.9, python-dotenv 1.2.4,
tokenizers 0.22.2, instalado em `.venv-retrieval`. `pip check` passou; todos os
scripts PowerShell passaram no parser; `git diff --check` passou.

Testes automatizados: **76 testes aprovados** ao executar
`scripts/retrieval/test.ps1` (7,428 segundos nesta execução). A suíte cobre
preparação, isolamento de gabaritos, fontes/aliases, tokenizer real com
vocabulários locais mínimos, Unicode, tamanhos em tokens, identidade,
persistência/reabertura em outro processo, retomada e CLI. Chroma é real nos
testes de índice, mas seus embeddings são determinísticos simulados. Os testes
HTTP e de comparação usam mocks. Esses testes não estabelecem qualidade de
recuperação nem disponibilidade do Ollama.

Execuções com os arquivos públicos reais:

| Verificação | Resultado |
| --- | --- |
| CLI `prepare` do corpus legal | Sucesso: 478 → 461; 715 perguntas |
| CLI `prepare --include-carf` em diretório separado | Sucesso: 7.682 → 7.665; 715 perguntas |
| Auditoria dos 7.682 registros de origem | Texto normalizado e filename preservados em cada documento canônico |
| Auditoria das referências | 598 perguntas com vínculos explícitos, 459 filenames, todos resolvidos; nenhuma referência explícita a CARF |
| Tokenização do corpus legal completo | Sucesso nos três tamanhos; 19.888 / 8.995 / 4.392 chunks; máximos de entrada 323 / 578 / 1089 tokens |
| Tokenização do corpus combinado completo, 512/50 | Sucesso: 7.665 documentos, 121.598 chunks, entrada máxima 578 tokens, sem falhas ou ajustes de overlap |
| Indexação real, corpus legal completo, 512/50 | Falhou ao consultar `/api/tags`: `WinError 10061`, conexão recusada no Ollama `http://localhost:11435` |
| Comparação real, cinco configurações, dez perguntas | 50 registros de erro de construção, sem busca executada; latências de busca `null` |
| Consulta real, corpus completo, 512/50 | Falha de abertura por índice ausente, registrada em JSONL e manifesto |

Não foi produzido embedding real ou resultado de recuperação bem-sucedido.
Nenhum resultado simulado foi usado para preencher as execuções reais.
Os detalhes de dados e tokens estão em [data-validation.md](data-validation.md)
e [token-validation.md](token-validation.md).

Logs locais, ignorados pelo Git:

- `.cache/retrieval-tests.log`, `.cache/real-prepare.json`,
  `.cache/real-prepare-carf.json`, `.cache/real-build.log`.
- Comparações: `retrieval_results/verification/20261031T030801_5df93e04/`.
  O manifesto raiz e os cinco subdiretórios registram configuração, corpus,
  modelo solicitado, estágios e erros; digest indisponível permanece `null`.
- Consulta: `retrieval_results/verification/20261031T030800_b80ff356/`.
- Manifestos preparados em `data/retrieval/prepared/manifest.json` e
  `data/retrieval-carf/prepared/manifest.json`.

## Comandos para reproduzir

Na raiz do repositório, em PowerShell:

```powershell
.\scripts\retrieval\setup.ps1
.\scripts\retrieval\test.ps1
$python = '.\.venv-retrieval\Scripts\python.exe'

# Normas completas e perguntas separadas.
& $python -m app.retrieval prepare
& $python scripts/retrieval/validate_tokens.py --data-dir data/retrieval

# Ollama deve estar ativo neste endpoint e ter o modelo instalado.
$env:OLLAMA_BASE_URL = 'http://localhost:11435'
# No Ollama do Compose já iniciado:
# docker exec ia025_ollama ollama pull embeddinggemma:300m
# Alternativa nativa: scripts/retrieval/start-local-ollama.ps1 -OllamaPath <ollama.exe> -PullModel

# Uma configuração de corpus completo; build pode ser repetido para retomar.
.\scripts\retrieval\build.ps1 -UsePrepared -Selection all -ChunkSize 512 -ChunkOverlap 50
.\scripts\retrieval\query.ps1 'Quem pode ser dependente no IRPF?' -Selection all -ChunkSize 512 -ChunkOverlap 50 -TopK 3 -OutputDir retrieval_results

# As cinco comparações, em três índices; retire -Limit para todas as perguntas.
.\scripts\retrieval\compare.ps1 -Cases data/retrieval/prepared/questions.jsonl -Limit 10 -OutputDir retrieval_results -Selection all -ChunkOverlap 50 -BuildIndexes

# CARF combinado sem substituir o corpus legal padrão.
& $python -m app.retrieval prepare --include-carf --data-dir data/retrieval-carf --raw-dir data/retrieval/raw/9f0f4263928c6152a40f888da3e63375a24bb0f3
.\scripts\retrieval\build.ps1 -UsePrepared -DataDir data/retrieval-carf -IndexDir data/retrieval-carf/chroma -Selection all -ChunkSize 512 -ChunkOverlap 50
.\scripts\retrieval\query.ps1 'Quem pode ser dependente no IRPF?' -DataDir data/retrieval-carf -IndexDir data/retrieval-carf/chroma -Selection all -ChunkSize 512 -ChunkOverlap 50 -TopK 3
```

Para verificação rápida, omita `-Selection all` em build/query/compare e use a
seleção existente de três normas. Para consultar 256 ou 1024 tokens, use o mesmo
`-ChunkSize` na construção e consulta, mantendo `-ChunkOverlap 50`. Alterar só
`-TopK` não exige construir novamente. Pela CLI Python, os argumentos
equivalentes são `--chunk-unit tokens --chunk-size <n> --chunk-overlap 50
--top-k <k> --selection all` (top-k pertence a query/compare).

Uso pelos colegas, após construir o índice correspondente:

```python
from dataclasses import replace
from app.retrieval import RetrievalConfig, Retriever

config = replace(RetrievalConfig.from_env(), corpus_selection=None,
                 chunk_unit="tokens", chunk_size=512, chunk_overlap=50)
chunks = Retriever(config).retrieve("Quem pode ser dependente no IRPF?", top_k=3)
contexts = [chunk["text"] for chunk in chunks]
sources = [chunk["source"] for chunk in chunks]
```

Os arquivos `retrieval.jsonl` também podem ser lidos diretamente por linha com
`json.loads`; `retrieved_chunks` preserva os metadados completos e
`retrieved_contexts` oferece a lista de textos. Esta entrega termina nesses
contratos de recuperação.

## Pendências

A execução real dos índices e das consultas/comparações permanece pendente de
Ollama ativo com `embeddinggemma:300m` no endpoint configurado. O corpus e o
tokenizer estão preparados; os comandos acima constroem ou retomam os índices
quando o serviço estiver disponível. Não há pendência de localização do CARF.

A escolha do segundo dataset depende do grupo. A [análise das três opções](second-dataset.md)
recomenda LARI para generalização fora do domínio tributário, JurisTCU quando
forem prioritários os julgamentos explícitos de relevância para recuperação, e
considera LegalBench.BR menos direto para RAG por não oferecer corpus separado
e qrels no esquema publicado. Nenhum deles foi ingerido.

## Arquivos alterados ou adicionados

| Parte | Arquivos |
| --- | --- |
| Configuração e documentação | `.env.example`, `.gitignore`, `README.md`, `docs/data-validation.md`, `docs/token-validation.md`, `docs/second-dataset.md`, `docs/member1-validation.md` |
| Dados e recuperação | `app/retrieval/data.py`, `chunking.py`, `tokenizer.py`, `config.py`, `embeddings.py`, `index.py`, `__main__.py`, `experiments.py`, `requirements.txt` (todos sob `app/retrieval/`) |
| Scripts | `scripts/retrieval/build.ps1`, `query.ps1`, `compare.ps1`, `validate_tokens.py` (todos sob `scripts/retrieval/`) |
| Novos testes | `tests/test_retrieval_carf.py`, `test_retrieval_tokens.py`, `test_retrieval_token_index.py`, `test_retrieval_experiments.py`, `test_retrieval_cli.py` (todos sob `tests/`) |
