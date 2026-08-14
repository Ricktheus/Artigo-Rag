"""
Configurações Globais do Sistema de RAG Hierárquico para Normas Regulamentadoras (NRs).
Centraliza caminhos, escopo de normas, parâmetros do Qdrant e modelos de Embedding/LLM.
"""

from pathlib import Path

# Diretórios base do projeto
BASE_DIR = Path(__file__).resolve().parent
DATA_DIRS = [BASE_DIR / "nrs_extraidas", BASE_DIR]
STORAGE_DIR = BASE_DIR / "storage"
QDRANT_DATA_DIR = BASE_DIR / "qdrant_data"

# Garantir criação dos diretórios necessários
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
QDRANT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Escopo estrito das 5 Normas Regulamentadoras para o Artigo Acadêmico
SCOPE_NRS = {
    "NR-05": "nr_05_tree.json",
    "NR-06": "nr_06_tree.json",
    "NR-10": "nr_10_tree.json",
    "NR-11": "nr_11_tree.json",
    "NR-35": "nr_35_tree.json",
}

# Configurações do Qdrant Vector Store
COLLECTION_NAME = "normas_regulamentadoras_hierarchical"
ENABLE_HYBRID_SEARCH = True
SPARSE_MODEL_NAME = "Qdrant/bm25"

# Configurações de Embeddings
# Modelo multilíngue de alta performance para a língua portuguesa
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Parâmetros de Recuperação (Retriever)
DEFAULT_TOP_K = 4
ALPHA_HYBRID = 0.5  # 0.5 pondera igualmente busca densa e esparsa

# Caminho para persistência do Docstore (armazenamento de nós completos e relacionamentos)
DOCSTORE_PATH = STORAGE_DIR / "docstore_nodes.json"
