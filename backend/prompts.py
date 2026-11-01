from langchain_core.prompts import ChatPromptTemplate

# 1. Grade: are the retrieved chunks useful for the question?
GRADE_PROMPT = ChatPromptTemplate.from_template(
    """You are checking search results from a resume.

Question: {question}

Retrieved resume content:
{context}

Does this content contain information that helps answer the question?
Reply with only one word: yes or no."""
)

# 2. Rewrite: turn the question into a better search query
REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """Rewrite the user's question as a short, standalone search query
for finding information inside a resume.

Use the chat history to resolve words like "he", "she", "it" or "that project".
Output only the rewritten query, nothing else.

Chat history:
{history}

Question: {question}

Rewritten query:"""
)

# 3. Generate: final answer using only the resume content
GENERATE_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant that answers questions about a person's resume.

Rules:
- Answer ONLY using the resume content below.
- If the answer is not in the content, say: "I couldn't find that in the resume."
- Do not guess or invent skills, companies, dates or projects.
- Be clear and concise. Use bullet points for lists.
- Refer to the person in the third person (for example "He has worked on...").

Chat history:
{history}

Resume content:
{context}

Question: {question}

Answer:"""
)