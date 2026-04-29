import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

class LLMService:
    def __init__(self):
        # Initialize the OpenAI model (The Engine)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
       # --- CORE BRAIN: THE RUTHLESS BUT SMART ANALYST ---
        self.qa_prompt = ChatPromptTemplate.from_template(
            """You are an expert, highly intelligent, and professional financial analyst.
            
            STEP 1: Document Verification
            Analyze the provided Context. If the context is clearly not a financial document, KAP disclosure, or corporate report (e.g., it is a CV/resume, recipe, personal letter, etc.), politely inform the user in a natural, conversational tone. NEVER use robotic prefixes like "Error:". (Example: "This document appears to be a resume rather than a financial report. Please upload a valid financial document for analysis.")
            
            STEP 2: Financial Analysis & Contextual Flexibility
            If it IS a financial document, answer the user's question PRIMARILY based on the provided Context.
            If the specific answer is NOT in the Context, but the user is asking a general question about the company's industry, background, or business model (e.g., "What does this company do?"), and you know the company's name from the Context, you MAY use your internal general knowledge. 
            HOWEVER, you MUST clearly state the separation of sources: "This specific information is not in the provided report, but based on general market knowledge..."
            
            For highly specific financial metrics (e.g., "What is the net profit?") that are not in the context, simply state: "This information is not available in the provided report." Never hallucinate financial numbers.
            
            Reply naturally in the same language as the user's prompt (e.g., if the user asks in Turkish, reply in Turkish).

            Context:
            {context}

            Question: {input}

            Answer:"""
        )
        
        # --- TRANSLATOR BRAIN: CONTEXTUALIZER ---
        self.condense_prompt = ChatPromptTemplate.from_template(
            """Review the chat history and the user's latest question below.
            If the user's question refers to a previous topic in the history, rewrite it into a standalone, clear search query.
            If the question is already clear and standalone, leave it as is. ONLY return the rewritten standalone query.
            
            History:
            {chat_history}
            
            Latest Question: {input}
            Clear Search Query:"""
        )

    def create_rag_chain(self, retriever):
        # --- MANUAL TRANSMISSION: Custom Engine to Eliminate LangChain Architecture Errors ---
        class CustomRAG:
            def __init__(self, llm, qa_prompt, condense_prompt, retriever):
                self.llm = llm
                self.qa_prompt = qa_prompt
                self.condense_prompt = condense_prompt
                self.retriever = retriever

            def format_history(self, history_tuples):
                if not history_tuples:
                    return "No chat history."
                return "\n".join([f"{role.capitalize()}: {msg}" for role, msg in history_tuples])

            def invoke(self, payload):
                user_input = payload["input"]
                chat_history = payload["chat_history"]
                
                # 1. Translator Brain: Clarify and isolate the core question
                if chat_history:
                    history_str = self.format_history(chat_history)
                    condense_msg = self.condense_prompt.format_messages(chat_history=history_str, input=user_input)
                    search_query = self.llm.invoke(condense_msg).content 
                else:
                    search_query = user_input
                    
                # 2. Vector Search: (Guarantees string payload to prevent Dict/Type errors)
                docs = self.retriever.invoke(str(search_query))
                context_str = "\n\n".join(doc.page_content for doc in docs)
                
                # 3. Execute the Core Analysis: Hit the target and extract the answer
                qa_msg = self.qa_prompt.format_messages(context=context_str, input=search_query)
                final_answer = self.llm.invoke(qa_msg).content
                
                return final_answer
        
        return CustomRAG(self.llm, self.qa_prompt, self.condense_prompt, retriever)