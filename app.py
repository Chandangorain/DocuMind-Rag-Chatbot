import uuid
import streamlit as st

from backend.ingest import ingest_pdf, clear_session
from backend.retriever import get_retriever
from backend.graph import build_graph, ask

st.set_page_config(page_title=" RAG Assistant", page_icon="📄")
st.title("📄 RAG Assistant")
st.caption("Upload a document and ask questions about it.")


# ---------- Session state (survives Streamlit reruns) ----------
def init_state():
    st.session_state.session_id = uuid.uuid4().hex[:8]
    st.session_state.messages = []      # [{"role", "content", "sources"}]
    st.session_state.graph = None       # built after upload
    st.session_state.files = []         # names of processed files


if "session_id" not in st.session_state:
    init_state()
    st.session_state.uploader_key = 0   # changing this resets the uploader widget


# ---------- Sidebar: upload ----------
with st.sidebar:
    st.header("1. Upload")
    uploaded = st.file_uploader(
        "Resume PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if st.button("Process", type="primary", disabled=not uploaded):
        try:
            with st.spinner("Reading and indexing..."):
                for f in uploaded:
                    ingest_pdf(f, st.session_state.session_id)
                retriever = get_retriever(st.session_state.session_id)
                st.session_state.graph = build_graph(retriever)
                st.session_state.files = [f.name for f in uploaded]
            st.success("Ready! Ask your questions.")
        except ValueError as e:        # our own validation errors
            st.error(str(e))
        except Exception as e:
            st.error(f"Something went wrong: {e}")

    if st.session_state.files:
        st.write("**Loaded:**")
        for name in st.session_state.files:
            st.write(f"- {name}")

    if st.button("Upload new document"):
        if st.session_state.graph is not None:
            try:
                clear_session(st.session_state.session_id)
            except Exception:
                pass
        init_state()
        st.session_state.uploader_key += 1
        st.rerun()


# ---------- Chat ----------
ready = st.session_state.graph is not None

if not ready:
    st.info("👈 Upload a  PDF in the sidebar and click **Process** to start.")

# show previous messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.write(f"- {s}")

# new question
question = st.chat_input("Ask about the resume...", disabled=not ready)

if question:
    history = list(st.session_state.messages)   # history BEFORE this question
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                answer, sources = ask(st.session_state.graph, question, history)
        except Exception as e:
            answer, sources = f"Sorry, something went wrong: {e}", []

        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.write(f"- {s}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )