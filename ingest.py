"""
Script de Ingestão e Indexação Híbrida (Passos 2 e 3).
Executa:
1. Leitura e estruturação dos nós das 5 NRs do escopo (NR-05, NR-06, NR-10, NR-11, NR-35).
2. Construção de relacionamentos hierárquicos e metadados.
3. Inicialização do Qdrant local (./qdrant_data) com busca híbrida (Dense + Sparse BM25).
4. Geração de embeddings vetoriais e inserção na collection.
5. Persistência do Docstore local para resolução de hierarquia e referências cruzadas.
"""

import sys
import time
from pathlib import Path
from typing import List

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from qdrant_client import QdrantClient
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import config
from node_builder import build_all_scope_nodes, save_nodes_to_docstore


def setup_embedding_model() -> HuggingFaceEmbedding:
    """
    Configura o modelo de embedding open-source multilíngue.
    """
    print(f"[EMBEDDING] Carregando modelo local: {config.EMBEDDING_MODEL_NAME}...")
    embed_model = HuggingFaceEmbedding(
        model_name=config.EMBEDDING_MODEL_NAME,
        embed_batch_size=32
    )
    Settings.embed_model = embed_model
    return embed_model


def get_qdrant_vector_store(client: QdrantClient) -> QdrantVectorStore:
    """
    Inicializa o QdrantVectorStore com suporte a busca híbrida (Denso + BM25 esparso).
    """
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=config.COLLECTION_NAME,
        enable_hybrid=config.ENABLE_HYBRID_SEARCH,
        fastembed_sparse_model=config.SPARSE_MODEL_NAME if config.ENABLE_HYBRID_SEARCH else None,
    )
    return vector_store


def run_ingestion() -> None:
    """
    Pipeline principal de ingestão e indexação.
    """
    start_time = time.time()
    print("=" * 70)
    print(" INICIANDO PIPELINE DE INGESTÃO E INDEXAÇÃO HÍBRIDA (NRS BRASIL)")
    print("=" * 70)

    # 1. Carregar nós estruturados das 5 NRs
    nodes, lookup = build_all_scope_nodes()
    if not nodes:
        print("[ERRO] Nenhum nó foi carregado. Verifique os arquivos JSON no diretório.")
        sys.exit(1)

    # 2. Salvar cache do Docstore para uso do Retriever
    save_nodes_to_docstore(nodes, config.DOCSTORE_PATH)

    # 3. Configurar Embeddings
    embed_model = setup_embedding_model()

    # 4. Inicializar Qdrant Local em Disco
    print(f"[QDRANT] Conectando ao Qdrant local em: {config.QDRANT_DATA_DIR.resolve()}...")
    client = QdrantClient(path=str(config.QDRANT_DATA_DIR))
    vector_store = get_qdrant_vector_store(client)

    # 5. Criar StorageContext e VectorStoreIndex
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(f"[INDEXAÇÃO] Indexando {len(nodes)} nós com embeddings densos e esparsos (BM25)...")
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"[SUCESSO] Ingestão concluída com êxito em {elapsed:.2f} segundos!")
    print(f"[INFO] Coleção '{config.COLLECTION_NAME}' pronta no Qdrant.")
    print(f"[INFO] Dados persistidos em: {config.QDRANT_DATA_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    run_ingestion()
