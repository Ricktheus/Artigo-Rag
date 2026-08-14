"""
Pipeline de Geração de Resposta e Query Engine (Passo 5).
Integra o HierarchicalCrossReferenceRetriever com LLMs (Mock, Local ou OpenAI)
e formata o prompt especializado para análise jurídica e regulatória das NRs.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from llama_index.core import Settings, PromptTemplate
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.schema import NodeWithScore
from llama_index.core.base.response.schema import Response
from llama_index.core.llms import LLM
from llama_index.core.llms.mock import MockLLM

from retriever import HierarchicalCrossReferenceRetriever, ExpandedContextItem
import config


# Prompt especializado para o artigo acadêmico
NR_SYSTEM_PROMPT = """Você é um Engenheiro de Segurança do Trabalho e Jurista especialista em Normas Regulamentadoras do Brasil (NR-05, NR-06, NR-10, NR-11 e NR-35).

Sua tarefa é responder à pergunta do usuário utilizando ESTRITAMENTE o contexto hierárquico e as referências cruzadas fornecidas abaixo.

Instruções para Resposta Acadêmica e Técnica:
1. **Estrutura Hierárquica**: Diferencie a Regra Geral (Seção/Capítulo Pai) das Exigências Específicas e Exceções (Itens/Subitens Filhos).
2. **Citação Obrigatória**: Cite expressamente as Normas (ex: NR-10, NR-06) e a numeração exata de cada item (ex: 10.2.4, 6.5.1).
3. **Referências Cruzadas**: Caso o contexto contenha referências cruzadas entre normas (ex: uma exigência de NR-10 que demanda EPIs da NR-06 ou trabalho em altura da NR-35), explique a integração entre elas.
4. **Fidelidade Normativa**: Não invente regras; atenha-se ao texto legal extraído.

---------------------
CONTEXTO RECUPERADO (HIERARQUIA E REFERÊNCIAS CRUZADAS):
{context_str}
---------------------

PERGUNTA: {query_str}

RESPOSTA TÉCNICA FUNDAMENTADA:"""

PROMPT_TEMPLATE = PromptTemplate(NR_SYSTEM_PROMPT)


class SimpleAcademicSynthesizer:
    """
    Sintetizador acadêmico local para demonstração imediata sem custos de API externa.
    Realiza a compilação estruturada dos achados hierárquicos e normativos.
    """
    def synthesize(self, query_str: str, expanded_items: List[ExpandedContextItem]) -> str:
        if not expanded_items:
            return "Nenhuma norma regulamentadora relevante foi encontrada no escopo das 5 NRs para esta consulta."

        consulted_nrs = set()
        citations = []
        cross_refs_found = []

        lines = []
        lines.append(f"### Parecer Tecnico-Regulatorio sobre: \"{query_str}\"\n")
        lines.append("**1. Fundamentacao Normativa e Hierarquia:**")

        for idx, item in enumerate(expanded_items, 1):
            meta = item.primary_node.metadata
            doc = meta.get("document_id", "NR")
            chap = meta.get("chapter_number", "")
            path = meta.get("chapter_path", "")
            consulted_nrs.add(doc)
            citations.append(f"{doc} (Item {chap})")

            parent_info = ""
            if item.parent_node:
                p_meta = item.parent_node.metadata
                p_num = p_meta.get("chapter_number", "")
                p_title = p_meta.get("title", "")
                parent_info = f" (sob a regra geral da Secao {p_num}: *{p_title}*)" if p_title else f" (sob a Secao {p_num})"

            lines.append(f"- **Dispositivo Principal {idx} [{doc} - Item {chap}]:**{parent_info}")
            lines.append(f"  *Hierarquia:* `{path}`")
            lines.append(f"  *Texto Normativo:* \"{item.primary_node.text.strip()}\"\n")

            if item.cross_references:
                for ref in item.cross_references:
                    target_doc = ref.get("target_document")
                    target_item = ref.get("target_item")
                    ref_content = ref.get("content")
                    consulted_nrs.add(target_doc)
                    cross_refs_found.append(f"{doc} -> {target_doc} ({target_item})")
                    lines.append(f"  ↳ **Articulacao com {target_doc} (Item {target_item}):**")
                    lines.append(f"    *Exigencia Complementar:* \"{ref_content}\"\n")

        lines.append("**2. Conclusao e Aplicacao Pratica:**")
        lines.append(
            f"A analise integrada das normas ({', '.join(sorted(consulted_nrs))}) demonstra a necessidade de cumprimento "
            f"conjunto tanto das regras gerais de gestao de seguranca quanto dos dispositivos especificos citados ({', '.join(citations)})."
        )
        if cross_refs_found:
            lines.append(f"Destaca-se a interconexao regulamentar identificada: {'; '.join(cross_refs_found)}.")

        return "\n".join(lines)


class NRHierarchicalQueryEngine(CustomQueryEngine):
    """
    Query Engine customizado que combina o HierarchicalCrossReferenceRetriever
    com o mecanismo de síntese e formatação acadêmica.
    """
    retriever: HierarchicalCrossReferenceRetriever
    llm: Optional[LLM] = None
    use_academic_synthesizer: bool = True

    def __init__(
        self,
        retriever: HierarchicalCrossReferenceRetriever,
        llm: Optional[LLM] = None,
        use_academic_synthesizer: bool = False,
    ):
        super().__init__(
            retriever=retriever,
            llm=llm,
            use_academic_synthesizer=use_academic_synthesizer
        )

    def custom_query(self, query_str: str) -> Response:
        """
        Executa o pipeline completo:
        1. Recuperação Híbrida + Expansão Hierárquica + Resolução de Referências
        2. Injeção no Prompt do LLM ou Sintetizador Acadêmico
        3. Geração da Resposta Estruturada
        """
        # Obter itens expandidos para inspeção detalhada
        expanded_items = self.retriever.retrieve_expanded(query_str)
        nodes_with_score = self.retriever.retrieve(query_str)

        # Montar contexto concatenado
        context_parts = [item.to_formatted_context() for item in expanded_items]
        full_context_str = "\n\n" + ("=" * 50) + "\n\n".join(context_parts)

        # Se houver LLM real configurado (ex: OpenAI / Groq / Ollama)
        if self.llm and not isinstance(self.llm, MockLLM) and not self.use_academic_synthesizer:
            formatted_prompt = PROMPT_TEMPLATE.format(
                context_str=full_context_str,
                query_str=query_str
            )
            llm_output = self.llm.complete(formatted_prompt)
            response_text = str(llm_output)
        else:
            # Sintetizador Estruturado Acadêmico
            synthesizer = SimpleAcademicSynthesizer()
            response_text = synthesizer.synthesize(query_str, expanded_items)

        return Response(
            response=response_text,
            source_nodes=nodes_with_score,
            metadata={
                "expanded_items_count": len(expanded_items),
                "consulted_nrs": list(set(
                    item.primary_node.metadata.get("document_id") for item in expanded_items
                )),
                "raw_context": full_context_str
            }
        )


def build_query_engine(
    retriever: Optional[HierarchicalCrossReferenceRetriever] = None,
    openai_api_key: Optional[str] = None,
    use_mock: bool = False
) -> NRHierarchicalQueryEngine:
    """
    Factory function para instanciar o Query Engine configurado.
    """
    if retriever is None:
        retriever = HierarchicalCrossReferenceRetriever()

    llm = None
    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")

    if api_key and not use_mock:
        try:
            from llama_index.llms.openai import OpenAI
            llm = OpenAI(model="gpt-4o-mini", api_key=api_key)
            use_academic_synthesizer = False
            print("[LLM] Utilizando OpenAI (gpt-4o-mini) para geracao.")
        except Exception as e:
            print(f"[LLM] OpenAI indisponivel ({e}). Usando sintetizador academico local.")
            use_academic_synthesizer = True
    else:
        # Modo local / mock gratuito sem dependência de API
        use_academic_synthesizer = True
        llm = MockLLM()
        print("[LLM] Modo local/sintetizador academico ativo (custo zero, 100% offline).")

    return NRHierarchicalQueryEngine(
        retriever=retriever,
        llm=llm,
        use_academic_synthesizer=use_academic_synthesizer
    )
