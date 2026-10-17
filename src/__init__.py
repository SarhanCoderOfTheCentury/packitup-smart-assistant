"""PackItUp Smart Shopping Assistant — Multi-Agent LangGraph & ChromaDB RAG Package.

Exports LangChain & LangGraph primitives (using installed `langchain_core` and `langgraph`
in production, with a lightweight compatible fallback for minimal test environments).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

__version__ = "1.0.0"

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
    from langchain_core.tools import tool
except ImportError:
    @dataclass
    class Document:  # type: ignore[no-redef]
        page_content: str
        metadata: Dict[str, Any] = field(default_factory=dict)

    class Embeddings:  # type: ignore[no-redef]
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            raise NotImplementedError

        def embed_query(self, text: str) -> List[float]:
            raise NotImplementedError

    @dataclass
    class BaseMessage:  # type: ignore[no-redef]
        content: str

    class HumanMessage(BaseMessage):  # type: ignore[no-redef]
        pass

    class AIMessage(BaseMessage):  # type: ignore[no-redef]
        pass

    class SystemMessage(BaseMessage):  # type: ignore[no-redef]
        pass

    def tool(name_or_fn: Any = None, args_schema: Any = None):  # type: ignore[no-redef]
        def decorator(fn: Callable):
            fn.tool_name = name_or_fn if isinstance(name_or_fn, str) else fn.__name__
            fn.args_schema = args_schema
            fn.invoke = lambda kwargs: fn(**kwargs) if isinstance(kwargs, dict) else fn(kwargs)
            return fn

        if callable(name_or_fn):
            return decorator(name_or_fn)
        return decorator


try:
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
except ImportError:
    START = "__start__"
    END = "__end__"

    def add_messages(left: List[Any], right: List[Any]) -> List[Any]:  # type: ignore[no-redef]
        return (left or []) + (right or [])

    class _CompiledStateGraph:
        def __init__(self, nodes, edges, cond_edges):
            self.nodes = nodes
            self.edges = edges
            self.cond_edges = cond_edges

        def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
            current_state = dict(state)
            curr = self.edges.get(START)
            steps = 0
            while curr and curr != END and steps < 25:
                steps += 1
                node_fn = self.nodes[curr]
                patch = node_fn(current_state)
                if isinstance(patch, dict):
                    for k, v in patch.items():
                        if k == "messages" and "messages" in current_state:
                            current_state["messages"] = list(current_state["messages"]) + list(v)
                        else:
                            current_state[k] = v
                if curr in self.cond_edges:
                    router_fn, mapping = self.cond_edges[curr]
                    route_key = router_fn(current_state)
                    curr = mapping.get(route_key, END)
                else:
                    curr = self.edges.get(curr, END)
            return current_state

    class StateGraph:  # type: ignore[no-redef]
        def __init__(self, state_schema: Any) -> None:
            self.state_schema = state_schema
            self.nodes: Dict[str, Callable] = {}
            self.edges: Dict[str, str] = {}
            self.cond_edges: Dict[str, Any] = {}

        def add_node(self, name: str, fn: Callable) -> None:
            self.nodes[name] = fn

        def add_edge(self, src: str, dst: str) -> None:
            self.edges[src] = dst

        def add_conditional_edges(self, src: str, router_fn: Callable, mapping: Dict[str, str]) -> None:
            self.cond_edges[src] = (router_fn, mapping)

        def compile(self) -> _CompiledStateGraph:
            return _CompiledStateGraph(self.nodes, self.edges, self.cond_edges)
