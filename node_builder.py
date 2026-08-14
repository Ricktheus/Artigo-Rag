"""
Módulo de Construção e Estruturação de Nós (LlamaIndex TextNodes).
Processa os JSONs das Normas Regulamentadoras e reconstrói as relações
hierárquicas (PARENT, CHILD, PREVIOUS, NEXT) e os metadados de referência.
Utiliza UUIDs determinísticos para total compatibilidade com o Qdrant Local.
"""

import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
import config


def generate_node_uuid(node_id: str) -> str:
    """
    Gera um UUID v5 determinístico a partir do identificador da norma/item (ex: 'NR10_10.2.4').
    Garante compatibilidade total com os requisitos de ID do Qdrant (UUID ou inteiro).
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, node_id))


def find_json_file(doc_id: str, filename: str) -> Optional[Path]:
    """
    Busca o arquivo JSON da norma nos diretórios configurados.
    """
    for d in config.DATA_DIRS:
        candidate = d / filename
        if candidate.exists():
            return candidate
    return None


def derive_parent_chapter_number(chapter_num: str) -> Optional[str]:
    """
    Deriva o identificador do capítulo pai a partir do número do item.
    Exemplos:
        '10.2.1' -> '10.2'
        '10.2.1.1' -> '10.2.1'
        '10.2' -> '10' (se existir) ou None
        'ANEXO I' -> None
    """
    if not chapter_num or "." not in chapter_num:
        return None
    
    parts = chapter_num.split(".")
    if len(parts) > 1:
        return ".".join(parts[:-1])
    return None


def build_text_for_node(node_dict: dict) -> str:
    """
    Formata o texto principal do nó garantindo que títulos e conteúdos
    sejam representados de forma semanticamente rica para os modelos de embedding.
    """
    title = (node_dict.get("title") or "").strip()
    content = (node_dict.get("content") or "").strip()
    chapter_number = node_dict.get("chapter_number", "")
    doc_id = node_dict.get("document_id", "")

    if title and content:
        return f"[{doc_id} - Item {chapter_number}: {title}]\n{content}"
    elif title and not content:
        return f"[{doc_id} - Seção {chapter_number}: {title}]"
    elif content:
        return f"[{doc_id} - Item {chapter_number}]\n{content}"
    else:
        return f"[{doc_id} - Item {chapter_number}]"


def build_nodes_from_json(json_path: Path) -> List[TextNode]:
    """
    Lê o JSON estrutural de uma NR e cria os objetos TextNode com seus
    metadados e relacionamentos estruturais completos.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_nodes = data.get("nodes", [])
    text_nodes: List[TextNode] = []
    natural_id_lookup: Dict[str, TextNode] = {}

    doc_id_default = data.get("document_id", "NR")
    
    # 1ª Passagem: Instanciação dos TextNodes com metadados e UUIDs determinísticos
    for raw in raw_nodes:
        natural_node_id = raw.get("node_id")
        node_uuid = generate_node_uuid(natural_node_id)
        
        doc_id = raw.get("document_id") or doc_id_default
        chapter_number = str(raw.get("chapter_number", ""))
        depth = int(raw.get("depth", 0))
        chapter_path = raw.get("chapter_path", "")
        title = raw.get("title")
        content = raw.get("content", "")
        has_table = bool(raw.get("has_table", False))
        references = raw.get("references", [])

        node_text = build_text_for_node(raw)

        metadata = {
            "node_id": natural_node_id,
            "document_id": doc_id,
            "chapter_number": chapter_number,
            "chapter_path": chapter_path,
            "depth": depth,
            "title": title or "",
            "has_table": has_table,
            "references": references,
        }

        # Criar TextNode do LlamaIndex com UUID válido para o Qdrant
        text_node = TextNode(
            text=node_text,
            id_=node_uuid,
            metadata=metadata,
            excluded_embed_metadata_keys=["references", "has_table"],
            excluded_llm_metadata_keys=["has_table"],
        )

        # Inicializar lista de relacionamentos
        text_node.relationships[NodeRelationship.CHILD] = []

        text_nodes.append(text_node)
        natural_id_lookup[natural_node_id] = text_node

    # 2ª Passagem: Estabelecer relacionamentos estruturais (PARENT, CHILD, PREV, NEXT)
    doc_prefix = text_nodes[0].metadata["document_id"].replace("-", "") if text_nodes else ""

    for i, node in enumerate(text_nodes):
        chapter_num = node.metadata.get("chapter_number", "")
        
        # Conectar nó anterior (PREVIOUS) e próximo (NEXT) no mesmo documento
        if i > 0:
            node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                node_id=text_nodes[i - 1].node_id
            )
        if i < len(text_nodes) - 1:
            node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                node_id=text_nodes[i + 1].node_id
            )

        # Conectar Pai (PARENT) e Filho (CHILD)
        parent_chap = derive_parent_chapter_number(chapter_num)
        if parent_chap:
            parent_natural_id = f"{doc_prefix}_{parent_chap}"
            if parent_natural_id in natural_id_lookup:
                parent_node = natural_id_lookup[parent_natural_id]
                node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                    node_id=parent_node.node_id
                )
                
                # Adicionar este nó à lista de filhos do pai
                child_rel = parent_node.relationships.get(NodeRelationship.CHILD)
                if isinstance(child_rel, list):
                    child_rel.append(RelatedNodeInfo(node_id=node.node_id))
                elif child_rel is None:
                    parent_node.relationships[NodeRelationship.CHILD] = [
                        RelatedNodeInfo(node_id=node.node_id)
                    ]

    return text_nodes


def build_all_scope_nodes() -> Tuple[List[TextNode], Dict[str, TextNode]]:
    """
    Processa APENAS as 5 Normas Regulamentadoras definidas no escopo acadêmico:
    NR-05, NR-06, NR-10, NR-11 e NR-35.
    Ignora quaisquer outras NRs presentes no workspace.
    """
    all_nodes: List[TextNode] = []
    global_lookup: Dict[str, TextNode] = {}

    for doc_id, filename in config.SCOPE_NRS.items():
        json_file = find_json_file(doc_id, filename)
        if not json_file:
            print(f"[AVISO] Arquivo {filename} para {doc_id} não encontrado nos diretórios.")
            continue

        print(f"[PROCESSANDO] Lendo nós de {doc_id} a partir de: {json_file.name}")
        nodes = build_nodes_from_json(json_file)
        print(f" -> {len(nodes)} nós estruturados com sucesso para {doc_id}.")
        
        all_nodes.extend(nodes)
        for n in nodes:
            # Indexa tanto por UUID quanto por ID natural (ex: "NR10_10.2.4")
            global_lookup[n.node_id] = n
            natural_id = n.metadata.get("node_id")
            if natural_id:
                global_lookup[natural_id] = n

    print(f"\n[TOTAL] {len(all_nodes)} nós estruturados no escopo das 5 NRs.")
    return all_nodes, global_lookup


def save_nodes_to_docstore(nodes: List[TextNode], output_path: Path) -> None:
    """
    Serializa os nós e seus metadados/relacionamentos em um arquivo JSON local
    para permitir resolução instantânea no Retriever sem depender de chamadas remotas.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for n in nodes:
        node_dict = n.to_dict()
        serializable.append(node_dict)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"[DOCSTORE] {len(nodes)} nós salvos em cache local: {output_path}")


def load_nodes_from_docstore(input_path: Path) -> Dict[str, TextNode]:
    """
    Carrega o mapa de nós previamente estruturados a partir do cache JSON.
    Indexa por UUID e por identificador natural para buscas rápidas.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Docstore não encontrado em: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        raw_list = json.load(f)

    lookup: Dict[str, TextNode] = {}
    for d in raw_list:
        node = TextNode.from_dict(d)
        lookup[node.node_id] = node
        natural_id = node.metadata.get("node_id")
        if natural_id:
            lookup[natural_id] = node
    return lookup
