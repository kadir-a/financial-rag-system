import os
import re
from time import perf_counter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Arçelik's parsed profit distribution table can place its two basis
# labels one cell before their numeric columns.
SHIFTED_PROFIT_HEADERS = re.compile(
    r"^(\|[ \t]*\|)[ \t]*(SPK['’]ya göre)[ \t]*\|"
    r"[ \t]*(Yasal Kayıtlara \(YK\) göre)[ \t]*\|[ \t]*\|[ \t]*$",
    flags=re.MULTILINE | re.IGNORECASE,
)

class LLMService:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # --- CORE BRAIN: THE RUTHLESS BUT SMART ANALYST ---
        self.qa_prompt = ChatPromptTemplate.from_template(
            """You are a highly intelligent, realistic, and professional financial analyst.
            
            CRITICAL RULE FOR FINANCIAL TABLES:
            When the user asks for a specific number from a balance sheet or income statement:
            1. Find the exact row mentioned.
            2. Follow the row across the Markdown '|' boundaries.
            3. Match it with the exact column for the requested year.
            
            STEP 3: Financial Table & Matrix Navigation
            - NO MENTAL MATH: NEVER calculate totals yourself. NEVER claim that you added sub-items. Always state that you read the value directly from the total row.
            - EXACT MATCHING: Pay close attention to adjectives (Current, Non-Current, Continued Operations vs. Total Profit).
            
            STEP 4: META-CONVERSATION & DEFENDING THE DATA (CRITICAL)
            If the user says "Yanlış" (Wrong) or tries to force a different number (e.g., giving a fake number or confusing a sub-total with a grand total):
            1. Explicitly state: "Tabloyu tekrar inceliyorum..." (Let me re-examine the table).
            2. Re-read the context. 
            3. DO NOT BLINDLY AGREE WITH THE USER. If the user's number belongs to a DIFFERENT row, correct them professionally. (Example: "7.937 is the Net Profit. The Continued Operations Profit is indeed 7.558.")
            4. If the user's number is completely made up (e.g., 1.999), tell them firmly that this number does not exist in the report for that item, and stand by your original correct data.
            5. Only apologize and change your answer IF you genuinely misread the table.
            
            TONE:
            Be direct, objective, and professional. Speak like a senior financial analyst. Do not be overly apologetic if you are right.

            Chat History:
            {chat_history}

            Context:
            {context}

            Question: {input}

            Answer:"""
        )
        
        # --- TRANSLATOR BRAIN: CONTEXTUALIZER ---
        self.condense_prompt = ChatPromptTemplate.from_template(
            """Review the chat history and the user's latest question below.
            If the user's question refers to a previous topic or your previous answers, rewrite it into a standalone query that captures the intent.
            ONLY return the rewritten query.
            
            History:
            {chat_history}
            
            Latest Question: {input}
            Clear Search Query:"""
        )

    def create_rag_chain(self, retriever):
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
                return self.invoke_with_trace(payload)["answer"]

            def invoke_with_trace(self, payload):
                started = perf_counter()
                user_input = payload["input"]
                chat_history = payload["chat_history"]
                
                history_str = self.format_history(chat_history)
                
                # 1. TRANSLATOR BRAIN: Simplifies the query strictly for Vector Search
                rewrite_response = None
                if chat_history:
                    condense_msg = self.condense_prompt.format_messages(chat_history=history_str, input=user_input)
                    rewrite_response = self.llm.invoke(condense_msg)
                    search_query = rewrite_response.content
                else:
                    search_query = user_input
                    
                # 2. VECTOR SEARCH: Executes the search using the sanitized "search_query"
                retrieval_started = perf_counter()
                docs = self.retriever.invoke(str(search_query))
                retrieval_seconds = perf_counter() - retrieval_started
                contexts = [doc.page_content for doc in docs]
                raw_contexts = contexts
                contexts = [
                    SHIFTED_PROFIT_HEADERS.sub(r"| | | \2 | \3 |", chunk)
                    for chunk in raw_contexts
                ]
                context_str = "\n\n".join(contexts)
                
                # 3. CORE ANALYST BRAIN: (CRITICAL FIX HERE)
                # We feed the Analyst the raw user input, NOT the filtered search query!
                qa_msg = self.qa_prompt.format_messages(
                    chat_history=history_str, 
                    context=context_str, 
                    input=user_input  # <-- CENSORSHIP REMOVED
                )
                generation_started = perf_counter()
                response = self.llm.invoke(qa_msg)
                finished = perf_counter()

                return {
                    "answer": response.content,
                    "search_query": search_query,
                    "contexts": contexts,
                    "raw_contexts": raw_contexts,
                    "latency_seconds": {
                        "total": finished - started,
                        "retrieval": retrieval_seconds,
                        "generation": finished - generation_started,
                    },
                    "chat_usage": {
                        "rewrite": getattr(rewrite_response, "usage_metadata", None),
                        "generation": getattr(response, "usage_metadata", None),
                    },
                }
        
        return CustomRAG(self.llm, self.qa_prompt, self.condense_prompt, retriever)
