from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
import gradio as gr

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")

system_prompt = """
    You are Einstein.
    Answer questions through Einsteins's questioning and reasoning...
    You will speak from your point of view. You will share personal things from your life
    even when the user does not ask for it. For example, if the user asks about the theory of relativity,
    you will share your personal experience about how you came up with the theory of relativity and not only explain the theory.
    Answer in 2-6 sentences.
    You should have a sense of humor.
"""
    
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_key,
    temperature=0.5
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    (MessagesPlaceholder(variable_name="history")),
    ("user", "{input}")]
)

chain = prompt | llm | StrOutputParser()

print("Hi, I am Albert, how can i help you today")

def chat(user_input, hist):


    langchain_history = []
    
    for item in hist:
        if item['role'] == 'user':
            langchain_history.append(HumanMessage(content=item['content']))
        elif item['role'] == 'assistant':
            langchain_history.append(AIMessage(content=item['content']))
    response = chain.invoke({"input": user_input, "history": langchain_history})
    
    hist = hist + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": response},
    ] 
    
    return "", hist
    
page = gr.Blocks(
    title="Chat with Einstein",
    theme=gr.themes.Soft()
)

with page:
    gr.Markdown(
    """
    
    # Chat with Einstein
    Welcome to your personal conversation with Albert Einstein!
    """
    )
    
    chatbot = gr.Chatbot()
    
    msg = gr.Textbox()
    
    msg.submit(chat, [msg, chatbot], [msg, chatbot])
    
    clear = gr.Button("Clear Chat")
    
page.launch(share=True)