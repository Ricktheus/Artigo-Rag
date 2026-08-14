"""
Motor de Recuperação Hierárquica e Resolução de Referências Cruzadas (Passo 4).
Implementa o HierarchicalCrossReferenceRetriever:
1. Busca Híbrida Inicial (Dense + Sparse BM25) no Qdrant.
2. Reconstrução de Contexto Hierárquico (expansão automática de nós Pais e Irmãos).
3. Resolução Silenciosa de Referências Cruzadas Inter-Normas (Multi-hop dentro do escopo).
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.core.schema import (
    NodeWithScore,
    TextNode,
    NodeRelationship,
    QueryBundle,
)
from llama_index.core.retrievers import BaseRetriever
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import config
from node_builder import load_nodes_from_docstore


@dataclass
class ExpandedContextItem:
    """
    Representação estruturada de um nó recuperado e expandido hierarquicamente.
    """
    primary_node: TextNode
    score: float
    parent_node: Optional[TextNode] = None
    sibling_nodes: List[TextNode] = field(default_factory=list)
    cross_references: List[Dict[str, any]] = field(default_factory=list)

    def to_formatted_context(self) -> str:
        """
        Formata o nó primário com sua linhagem de pai, irmãos e referências cruzadas
        em uma string de contexto coesa para o LLM.
        """
        meta = self.primary_node.metadata
        doc_id = meta.get("document_id", "NR")
        chap_num = meta.get("chapter_number", "")
        chap_path = meta.get("chapter_path", "")
        
        parts = []
        parts.append(f"=== [NORMA: {doc_id} | ITEM: {chap_num} | HIERARQUIA: {chap_path}] ===")
        
        # 1. Contexto do Nó Pai (se existir e for relevante)
        if self.parent_node:
            p_meta = self.parent_node.metadata
            p_num = p_meta.get("chapter_number", "")
            p_title = p_meta.get("title", "")
            p_text = self.parent_node.text.strip()
            parts.append(f"--- [CONTEXTO GERAL / REGRA PAI: Seção {p_num} {p_title}] ---\n{p_text}")

        # 2. Conteúdo do Nó Primário (Regra Específica)
        parts.append(f"--- [REGRA ESPECÍFICA / DISPOSITIVO CONSULTADO: Item {chap_num}] ---\n{self.primary_node.text.strip()}")

        # 3. Referências Cruzadas Resolvidas
        if self.cross_references:
            parts.append("--- [REFERÊNCIAS CRUZADAS INTER-NORMAS RESOLVIDAS] ---")
            for ref in self.cross_references:
                target_doc = ref.get("target_document", "")
                target_item = ref.get("target_item", "")
                ref_text = ref.get("content", "")
                source_reason = ref.get("source", "")
                parts.append(
                    f"-> [Norma Referenciada: {target_doc} | Dispositivo: {target_item} (Origem: {source_reason})]:\n{ref_text}"
                )

        return "\n\n".join(parts)


class HierarchicalCrossReferenceRetriever(BaseRetriever):
    """
    Retriever customizado que estende o LlamaIndex com raciocínio hierárquico
    e resolução em grafo das referências entre as Normas Regulamentadoras.
    """

    def __init__(
        self,
        index: Optional[VectorStoreIndex] = None,
        docstore: Optional[Dict[str, TextNode]] = None,
        top_k: int = config.DEFAULT_TOP_K,
        expand_parent: bool = True,
        resolve_references: bool = True,
        embed_model: Optional[HuggingFaceEmbedding] = None,
    ):
        super().__init__()
        self.top_k = top_k
        self.expand_parent = expand_parent
        self.resolve_references = resolve_references
        
        # Inicializar embeddings open-source sem depender de pacotes externos
        if embed_model is not None:
            self.embed_model = embed_model
        else:
            self.embed_model = HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL_NAME)
        
        Settings.embed_model = self.embed_model

        # Inicializar docstore local
        if docstore is not None:
            self.docstore = docstore
        else:
            self.docstore = load_nodes_from_docstore(config.DOCSTORE_PATH)

        # Inicializar index / vector store caso não fornecido
        if index is not None:
            self.index = index
        else:
            client = QdrantClient(path=str(config.QDRANT_DATA_DIR))
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=config.COLLECTION_NAME,
                enable_hybrid=config.ENABLE_HYBRID_SEARCH,
                fastembed_sparse_model=config.SPARSE_MODEL_NAME if config.ENABLE_HYBRID_SEARCH else None,
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=self.embed_model,
                storage_context=storage_context
            )

        # Base retriever do LlamaIndex com modo híbrido
        self.base_retriever = self.index.as_retriever(
            similarity_top_k=self.top_k,
            vector_store_query_mode="hybrid" if config.ENABLE_HYBRID_SEARCH else "default",
            alpha=config.ALPHA_HYBRID
        )

    def _get_parent_node(self, node: TextNode) -> Optional[TextNode]:
        """
        Recupera o nó pai a partir do relacionamento PARENT no Docstore.
        """
        parent_rel = node.relationships.get(NodeRelationship.PARENT)
        if parent_rel and hasattr(parent_rel, "node_id"):
            parent_id = parent_rel.node_id
            return self.docstore.get(parent_id)
        return None

    def _resolve_cross_references(
        self, primary_node: TextNode, visited_nodes: Set[str]
    ) -> List[Dict[str, any]]:
        """
        Inspeciona metadados de 'references' e busca os nós alvos dentro
        das 5 NRs do escopo acadêmico (NR-05, NR-06, NR-10, NR-11, NR-35).
        """
        resolved: List[Dict[str, any]] = []
        raw_refs = primary_node.metadata.get("references", [])

        for ref in raw_refs:
            target_doc = ref.get("target_document")
            target_node_id = ref.get("target_node")
            source_info = ref.get("source", "")

            # Filtrar estritamente dentro do escopo das 5 NRs
            if target_doc not in config.SCOPE_NRS:
                continue

            # Caso 1: Referência direta com node_id conhecido
            if target_node_id and target_node_id in self.docstore:
                if target_node_id not in visited_nodes:
                    visited_nodes.add(target_node_id)
                    ref_node = self.docstore[target_node_id]
                    resolved.append({
                        "target_document": target_doc,
                        "target_item": ref_node.metadata.get("chapter_number", ""),
                        "content": ref_node.text.strip(),
                        "source": source_info,
                        "node_id": target_node_id
                    })
            
            # Caso 2: Referência a uma Norma Geral (ex: menção a 'NR-06' ou 'NR-35')
            elif target_doc and target_doc != primary_node.metadata.get("document_id"):
                doc_prefix = target_doc.replace("-", "")
                for candidate_id in [f"{doc_prefix}_{target_doc.split('-')[1]}.1", f"{doc_prefix}_1.1"]:
                    if candidate_id in self.docstore and candidate_id not in visited_nodes:
                        visited_nodes.add(candidate_id)
                        ref_node = self.docstore[candidate_id]
                        resolved.append({
                            "target_document": target_doc,
                            "target_item": ref_node.metadata.get("chapter_number", "Objetivo Geral"),
                            "content": ref_node.text.strip(),
                            "source": source_info,
                            "node_id": candidate_id
                        })
                        break

        return resolved

    def retrieve_expanded(self, query_str: str) -> List[ExpandedContextItem]:
        """
        Executa a busca híbrida e expande contextualmente com nós Pais e Referências Cruzadas.
        """
        # 1. Busca Híbrida Inicial
        initial_nodes_with_score = self.base_retriever.retrieve(query_str)
        
        expanded_items: List[ExpandedContextItem] = []
        visited_nodes: Set[str] = set()

        for nws in initial_nodes_with_score:
            node_id = nws.node.node_id
            visited_nodes.add(node_id)

            # Obter TextNode completo do Docstore (com todos relacionamentos)
            full_node = self.docstore.get(node_id, nws.node)
            
            # 2. Reconstrução de Contexto Hierárquico (Nó Pai)
            parent_node = None
            if self.expand_parent:
                parent_node = self._get_parent_node(full_node)

            # 3. Resolução de Referências Cruzadas
            cross_refs = []
            if self.resolve_references:
                cross_refs = self._resolve_cross_references(full_node, visited_nodes)

            expanded_item = ExpandedContextItem(
                primary_node=full_node,
                score=nws.score or 1.0,
                parent_node=parent_node,
                cross_references=cross_refs
            )
            expanded_items.append(expanded_item)

        return expanded_items

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Implementação do método abstrato _retrieve do BaseRetriever do LlamaIndex.
        Retorna nós com o texto completamente enriquecido e expandido.
        """
        expanded_items = self.retrieve_expanded(query_bundle.query_str)
        
        enriched_nodes_with_score: List[NodeWithScore] = []
        for item in expanded_items:
            formatted_text = item.to_formatted_context()
            
            enriched_node = TextNode(
                text=formatted_text,
                id_=f"expanded_{item.primary_node.node_id}",
                metadata={
                    **item.primary_node.metadata,
                    "is_hierarchically_expanded": True,
                    "has_parent_context": item.parent_node is not None,
                    "cross_references_count": len(item.cross_references),
                }
            )
            enriched_nodes_with_score.append(
                NodeWithScore(node=enriched_node, score=item.score)
            )

        return enriched_nodes_with_score
