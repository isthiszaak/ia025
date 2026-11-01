# from langchain_core.prompts import PromptTemplate

# Define your chatbot prompt template
CHAT_TEMPLATE = """Your are a helpful assistant.
Chat history: {chat_history}
User question: {user_input}
Answer:"""

# Define your RAG prompt template
RAG_PROMPT = """You are a helpful assistant.
Answer the question using the provided context.

If the context does not contain enough information to answer,
clearly state that the available information is insufficient.
Do not invent facts.

Retrieved context:
{context}

User question:
{user_input}

Answer:"""

# chat_prompt_template = PromptTemplate(
#         input_variables=["chat_history", "user_input"],
#         template=CHAT_TEMPLATE
#     )