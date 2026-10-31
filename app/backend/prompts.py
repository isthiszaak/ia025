from langchain_core.prompts import PromptTemplate

# Define your chatbot prompt template
CHAT_TEMPLATE = """Your are a helpful assistant.
Chat history: {chat_history}
User question: {user_input}
Answer:"""

chat_prompt_template = PromptTemplate(
        input_variables=["chat_history", "user_input"],
        template=CHAT_TEMPLATE
    )