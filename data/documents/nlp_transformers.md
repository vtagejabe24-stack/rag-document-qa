# Natural Language Processing and Transformers

Natural language processing (NLP) is the field of building systems that understand and generate human language. Typical tasks include text classification, machine translation, question answering, summarization and named entity recognition.

## Tokenization and Embeddings

Tokenization splits text into units called tokens. Modern models use subword tokenization such as WordPiece or byte-pair encoding, so a rare word is broken into smaller pieces and the vocabulary stays manageable. Each token is then mapped to an embedding, a dense vector of numbers. Word2Vec and GloVe produce one fixed vector per word, so the word "bank" gets the same vector in "river bank" and in "bank account". Contextual embeddings from models such as BERT produce a different vector for each occurrence, depending on the surrounding words.

## Self-Attention

Self-attention lets every token in a sequence look at every other token and decide how much each one matters. Each token is projected into three vectors: a query, a key and a value. The attention weight between two tokens is the scaled dot product of the query of one and the key of the other, passed through a softmax. The output for a token is the weighted sum of the value vectors. Because attention has no built-in notion of order, positional encodings are added to the embeddings so the model knows where each token sits in the sequence. Multi-head attention runs several attention operations in parallel so the model can capture different kinds of relationships.

## The Transformer Architecture

The Transformer was introduced in the 2017 paper "Attention Is All You Need". It stacks layers that combine multi-head self-attention with a feed-forward network, wrapped with residual connections and layer normalization. Unlike RNNs, a Transformer processes all tokens of a sequence in parallel, which makes training much faster on GPUs and allows models to scale to billions of parameters. The original design has an encoder that reads the input and a decoder that generates the output.

## BERT, T5 and GPT

BERT is an encoder-only model. It is pretrained with masked language modeling, where random tokens are hidden and the model must predict them from both left and right context. This makes BERT well suited to classification and retrieval. GPT models are decoder-only and are trained to predict the next token, which makes them natural text generators. T5 is an encoder-decoder model that treats every task as text-to-text.

## Retrieval-Augmented Generation

Retrieval-augmented generation (RAG) combines a retriever with a language model. Documents are split into chunks, and each chunk is converted into an embedding and stored in a vector index such as FAISS. At question time the question is embedded, the most similar chunks are retrieved by cosine similarity, and they are placed in the prompt so the language model can answer from them. RAG reduces hallucination because the answer is grounded in retrieved text, and it lets a system use private or up-to-date documents without retraining the model. A cross-encoder reranker can re-score the retrieved chunks by reading the question and each chunk together, which improves the ordering at the cost of extra computation.

## Chunk Size

Chunk size is a trade-off. Small chunks make retrieval precise but may lose the surrounding context needed to answer. Large chunks keep more context but mix unrelated topics into one embedding, which blurs the match with the question. Overlap between consecutive chunks reduces the chance that an answer is cut in half at a boundary.
