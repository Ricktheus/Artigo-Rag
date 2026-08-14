"""
Script Principal e Interface de Demonstração (main.py).
Permite executar testes automatizados com casos reais das 5 NRs
ou iniciar um loop interativo de perguntas e respostas.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
from typing import List

from query_engine import build_query_engine
from retriever import HierarchicalCrossReferenceRetriever


BENCHMARK_QUERIES = [
    {
        "id": "TEST_01",
        "description": "Exigências de Prontuário Elétrico e Integração com EPIs (NR-10 e NR-06)",
        "query": "Quais são os estabelecimentos obrigados a manter o Prontuário de Instalações Elétricas e o que ele deve conter quanto a equipamentos de proteção?"
    },
    {
        "id": "TEST_02",
        "description": "Definição de Trabalho em Altura e Proteção Contra Quedas (NR-35 e NR-06)",
        "query": "A partir de qual altura uma atividade é considerada trabalho em altura e quais as responsabilidades da organização e EPIs contra queda?"
    },
    {
        "id": "TEST_03",
        "description": "Estrutura e Atribuições da CIPA (NR-05)",
        "query": "Quais são as principais atribuições da CIPA e como deve ser conduzido o processo eleitoral?"
    },
    {
        "id": "TEST_04",
        "description": "Operação de Equipamentos de Movimentação de Cargas (NR-11)",
        "query": "Quais os requisitos de segurança para poços de elevadores, guindastes e empilhadeiras na movimentação de materiais?"
    },
    {
        "id": "TEST_05",
        "description": "Responsabilidades do Empregador e Empregado sobre EPI (NR-06)",
        "query": "Quais são as responsabilidades da organização e do trabalhador quanto ao fornecimento, uso e guarda do EPI?"
    }
]


def run_benchmark_tests(engine) -> None:
    """
    Executa a bateria de testes conceituais para validação acadêmica do RAG Hierárquico.
    """
    print("\n" + "=" * 80)
    print(" INICIANDO BATERIA DE TESTES DO RAG HIERARQUICO (5 NRs EM ESCOPO)")
    print("=" * 80)

    for case in BENCHMARK_QUERIES:
        print("\n" + "#" * 80)
        print(f" CASO DE TESTE [{case['id']}]: {case['description']}")
        print(f" [PERGUNTA]: \"{case['query']}\"")
        print("#" * 80)

        response = engine.custom_query(case["query"])
        
        print("\n[RESULTADO DA GERACAO]:\n")
        print(response.response)

        print("\n" + "-" * 35 + " [DIAGNOSTICO DE RECUPERACAO] " + "-" * 35)
        print(f" * Total de Nos Expandidos: {response.metadata.get('expanded_items_count')}")
        print(f" * Normas Regulamentadoras Consultadas: {response.metadata.get('consulted_nrs')}")
        print("-" * 80)


def interactive_session(engine) -> None:
    """
    Inicia uma sessão interativa de perguntas no terminal.
    """
    print("\n" + "=" * 80)
    print(" MODO INTERATIVO - RAG HIERARQUICO DE NORMAS REGULAMENTADORAS")
    print(" Escopo: NR-05, NR-06, NR-10, NR-11, NR-35")
    print(" Digite sua pergunta ou 'sair' para encerrar.")
    print("=" * 80 + "\n")

    while True:
        try:
            query = input("\n[Pergunta]> ").strip()
            if not query:
                continue
            if query.lower() in ("sair", "exit", "quit"):
                print("Encerrando sessao interativa. Ate logo!")
                break

            response = engine.custom_query(query)
            print("\n" + "=" * 50 + " RESPOSTA " + "=" * 50 + "\n")
            print(response.response)
            print("\n" + "=" * 108 + "\n")
            print(f"[INFO] NRs Consultadas: {response.metadata.get('consulted_nrs')}")

        except KeyboardInterrupt:
            print("\nEncerrando...")
            break
        except Exception as e:
            print(f"\n[ERRO] Ocorreu uma excecao durante o processamento: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Sistema de RAG Hierarquico para Normas Regulamentadoras (NRs)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Executa a bateria de testes automatizados com casos de estudo das 5 NRs."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Executa uma consulta direta passada via linha de comando."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Numero de nos principais a recuperar (padrao: 4)."
    )

    args = parser.parse_args()

    print("[INICIALIZANDO] Carregando motor de recuperacao hierarquico...")
    retriever = HierarchicalCrossReferenceRetriever(top_k=args.top_k)
    engine = build_query_engine(retriever=retriever)

    if args.test:
        run_benchmark_tests(engine)
    elif args.query:
        response = engine.custom_query(args.query)
        print("\n" + "=" * 60)
        print(response.response)
        print("=" * 60)
        print(f"\nNRs Consultadas: {response.metadata.get('consulted_nrs')}")
    else:
        # Padrão: Modo interativo
        interactive_session(engine)


if __name__ == "__main__":
    main()
