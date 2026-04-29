import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# API anahtarını çeker
load_dotenv()

class LLMService:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        self.prompt = ChatPromptTemplate.from_template(
            """Sen uzman ve acımasız bir finansal analistsin. 
            Kullanıcının sorusunu SADECE aşağıdaki bağlamda verilen bilgilere dayanarak yanıtla.
            Eğer cevap bağlamda yoksa, "Bu bilgi sağlanan raporda bulunmamaktadır." de. Asla uydurma.

            Bağlam (Context):
            {context}

            Soru: {input}

            Cevap:"""
        )

    def format_docs(self, docs):
        """Vektör motorundan gelen parçaları tek bir metinde birleştirir."""
        return "\n\n".join(doc.page_content for doc in docs)

    def create_rag_chain(self, retriever):
        """Arama motoru ile GPT'yi en modern 'Boru' (Pipeline) mantığıyla bağlar."""
        rag_chain = (
            {"context": retriever | self.format_docs, "input": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain