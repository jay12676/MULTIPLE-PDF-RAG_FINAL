from langgraph.graph import END, StateGraph

from app.reasoning.state import QAState
from app.reasoning.nodes.query_decomposer import query_decomposer
from app.reasoning.nodes.retriever_node import retriever_node
from app.reasoning.nodes.aggregator_node import aggregator_node
from app.reasoning.nodes.reasoning_node import reasoning_node
from app.reasoning.nodes.formatter_node import formatter_node


def build_qa_graph():
    graph = StateGraph(QAState)

    graph.add_node("decompose", query_decomposer)
    graph.add_node("retrieve",  retriever_node)
    graph.add_node("aggregate", aggregator_node)
    graph.add_node("reason",    reasoning_node)
    graph.add_node("format",    formatter_node)

    graph.set_entry_point("decompose")

    graph.add_edge("decompose", "retrieve")
    graph.add_edge("retrieve",  "aggregate")
    graph.add_edge("aggregate", "reason")
    graph.add_edge("reason",    "format")
    graph.add_edge("format",    END)

    return graph.compile()


# Compiled graph — imported and used by the query endpoint
qa_chain = build_qa_graph()
