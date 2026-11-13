#retrieve → grade ─ yes ──────────────→ generate → END
#              │
#              └ no (and tries < MAX_REWRITES) → rewrite → back to retrieve

from typing import TypedDict, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from backend.config import LLM_MODEL, OPENAI_API_KEY, MAX_REWRITES
from backend.prompts import GRADE_PROMPT, REWRITE_PROMPT, GENERATE_PROMPT


# ---------- State: the data that flows through the graph ----------
class GraphState(TypedDict):
    question: str            # the user's original question
    search_query: str        # what we search with (may get rewritten)
    history: str             # earlier chat, as plain text
    docs: List[Document]     # retrieved chunks
    relevant: bool           # did the grader say yes?
    tries: int               # how many rewrites so far
    answer: str
    sources: List[str]


# Takes the retrieved chunks and joins them into one block of text, separated by blank lines. 
def format_context(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


def format_history(messages: list, last_n: int = 6) -> str:
    """messages = [{"role": "user"/"assistant", "content": "..."}]"""
    lines = [f"{m['role']}: {m['content']}" for m in messages[-last_n:]]
    return "\n".join(lines) if lines else "No previous messages."


# ---------- Graph ----------
def build_graph(retriever):
    llm = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY, temperature=0)
    parser = StrOutputParser()

    grade_chain = GRADE_PROMPT | llm | parser #prompts are stpred in prompts.py file
    rewrite_chain = REWRITE_PROMPT | llm | parser
    generate_chain = GENERATE_PROMPT | llm | parser

    # ----- Nodes -----
    def retrieve(state: GraphState):
        docs = retriever.invoke(state["search_query"])
        return {"docs": docs}   #Searches Chroma with the current query and stores the 4 best chunks in docs.

    def grade(state: GraphState):
        reply = grade_chain.invoke({
            "question": state["question"],
            "context": format_context(state["docs"]),
        })
        return {"relevant": "yes" in reply.lower()}

    def rewrite(state: GraphState):
        new_query = rewrite_chain.invoke({
            "question": state["question"],
            "history": state["history"],
        })
        return {"search_query": new_query.strip(), "tries": state["tries"] + 1}

    def generate(state: GraphState):
        answer = generate_chain.invoke({
            "question": state["question"],
            "context": format_context(state["docs"]),
            "history": state["history"],
        })
        sources = sorted({
            f"{d.metadata.get('source', 'resume')} (page {d.metadata.get('page', 0) + 1})"
            for d in state["docs"]
        })
        return {"answer": answer, "sources": sources}

    # ----- Decision: what happens after grading? -----
    def route_after_grade(state: GraphState):
        if state["relevant"] or state["tries"] >= MAX_REWRITES:
            return "generate"
        return "rewrite"

    # ----- Wire it together -----
    g = StateGraph(GraphState)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("rewrite", rewrite)
    g.add_node("generate", generate)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade",
        route_after_grade,
        {"generate": "generate", "rewrite": "rewrite"},
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)

    return g.compile()


# ---------- One simple function for the UI to call ----------
def ask(graph, question: str, messages: list):
    """Runs the graph. Returns (answer, sources)."""
    result = graph.invoke({
        "question": question,
        "search_query": question,
        "history": format_history(messages),
        "docs": [],
        "relevant": False,
        "tries": 0,
        "answer": "",
        "sources": [],
    })
    return result["answer"], result["sources"]