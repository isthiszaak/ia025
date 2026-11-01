# Validação do chunking em tokens — Member 1

O módulo mantém `chunk_documents(documents, size, overlap)` em caracteres para
compatibilidade. Para tokens, a chamada é
`chunk_documents(documents, size, overlap, chunk_unit="tokens", tokenizer=tokenizer)`.
O CLI usa `--chunk-unit tokens`; os valores previstos são 256, 512 e 1024.

## Tokenizer reproduzível e preservação

Usamos apenas o tokenizer de EmbeddingGemma, sem carregar pesos do modelo:

- Identificador: `onnx-community/embeddinggemma-300m-ONNX`.
- Revisão: `8cd82455fa7fae53af154775fd1494f5577ef4b0`.
- Arquivo: `tokenizer.json`, SHA256
  `4dda02faaf32bc91031dc8c88457ac272b00c1016cc679757d1c441b248b9c47`.
- Biblioteca: `tokenizers==0.22.2`. A versão efetiva, identificação, revisão e hash
  entram nos metadados dos chunks e na identidade dos índices.

O [arquivo na revisão fixa](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/blob/8cd82455fa7fae53af154775fd1494f5577ef4b0/tokenizer.json)
publica esse SHA256. O [cartão do modelo](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX)
identifica o modelo base Google, os prefixos de busca e o contexto de 2048 tokens.
O cache local verifica o hash a cada leitura; não há estimativa por caracteres.

Cada trecho é uma fatia exata do documento normalizado, sem usar o texto
decodificado como substituto. `start_char`/`end_char` usam posições Unicode Python;
`start_token`/`end_token` localizam a janela na tokenização do documento completo.
Espaços, quebras de linha, acentos e caracteres ignorados pelo normalizador do
tokenizer continuam no texto. Cada fatia é novamente tokenizada para confirmar
`token_count <= chunk_size`, pois cortar um documento pode alterar a tokenização
na borda. Identificadores incluem unidade, parâmetros, tokenizer, offsets e texto;
os IDs antigos em caracteres permanecem compatíveis.

O overlap configurado é 50 tokens nas execuções abaixo, escolhido para este
trabalho e configurável. Não é uma exigência do plano. Múltiplos tokens de
byte fallback podem corresponder ao mesmo caractere Unicode. Nessa situação,
um corte interno destruiria o texto; a borda é alinhada a um caractere completo.
`actual_overlap_tokens` registra a sobreposição efetiva, sem esconder o ajuste.
O auditor lista os chunks afetados. O valor configurado permanece constante entre
os tamanhos comparados.

O orçamento dos chunks exclui prefixos e tokens especiais. A entrada completa
de embedding é contada separadamente: `title: <título> | text: <trecho>`, mais
BOS/EOS do tokenizer. O cliente também mantém `truncate: false` no Ollama; títulos
ou perguntas grandes falham explicitamente em vez de serem cortados.

## Verificação real do corpus legal

Executada em 31/10/2026 com a revisão BR-TaxQA-R
`9f0f4263928c6152a40f888da3e63375a24bb0f3`, 461 documentos jurídicos deduplicados.
Estas são contagens reais do tokenizer; não são embeddings simulados nem uma
indexação no Ollama.

| Chunk / overlap em tokens | Chunks | Maior trecho em tokens | Maior entrada completa | Ajustes Unicode de overlap |
| --- | ---: | ---: | ---: | ---: |
| 256 / 50 | 19.888 | 256 | 323 | 1 |
| 512 / 50 | 8.995 | 512 | 578 | 0 |
| 1024 / 50 | 4.392 | 1024 | 1089 | 0 |

Todas as entradas ficaram abaixo de 2048 tokens. O auditor verifica cobertura
completa, ausência de lacunas, offsets, texto exato, documento/fonte e limites.
O JSON gerado inclui hash do corpus, tokenizer, contagens e eventuais falhas.
O único ajuste foi de 50 para 48 tokens, na Solução de Consulta Cosit nº 70/2013:
caracteres de uso privado presentes na fonte ocupam três tokens de byte fallback
cada. A auditoria completa terminou com zero falhas em 139,08 segundos nesta
máquina; esse tempo mede validação/tokenização local, não recuperação.

## Verificação do corpus combinado com CARF

O mesmo auditor verificou os 7.665 documentos do corpus combinado (461 normas +
7.204 acórdãos), com chunks de 512 e overlap de 50 tokens:

| Documentos verificados | Chunks | Maior trecho em tokens | Maior entrada completa | Ajustes de overlap | Falhas |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7.665 | 121.598 | 512 | 578 | 0 | 0 |

O relatório completo está no arquivo local ignorado
`data/retrieval-carf/token_validation.json`, identificado pelo hash do corpus e
pela revisão do tokenizer. Também aqui não foram gerados embeddings, executadas
buscas nem medidas de qualidade.

## Reproduzir sem embeddings

Na raiz do repositório, após preparar os dados:

```powershell
& '.\.venv-retrieval\Scripts\python.exe' scripts/retrieval/validate_tokens.py --data-dir data/retrieval

# Corpus combinado, preparado separadamente com --include-carf:
& '.\.venv-retrieval\Scripts\python.exe' scripts/retrieval/validate_tokens.py --data-dir data/retrieval-carf --sizes 512 --tokenizer-cache data/retrieval/tokenizer

# Testes unitários usam tokenizers reais com vocabulários locais mínimos.
& '.\.venv-retrieval\Scripts\python.exe' -m unittest discover -s tests -p test_retrieval_tokens.py -v
```

Os relatórios ficam em `<data-dir>/token_validation.json` por padrão. O auditor
aceita `--sizes`, `--overlap`, `--document-limit` e `--report`; relatórios com limite
de documentos identificam quantos documentos foram efetivamente verificados.
São 11 testes de tokens, cobrindo tamanhos do plano, Unicode/byte fallback,
espaços ignorados, tokens especiais literais, IDs, fontes, integridade do cache,
contagem de especiais e rejeição de revisões mutáveis.
