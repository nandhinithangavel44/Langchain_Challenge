#TASK 16: Conversational RAG
"""
------------------------------
Build a RAG pipeline that is aware of conversation history.


Requirements:
  - Use create_history_aware_retriever to rephrase follow-up
    questions into standalone queries.
  - Use create_retrieval_chain + create_stuff_documents_chain
    to answer with context.
  - Run a 2-turn conversation:
      Turn 1: "What is LangChain?"
      Turn 2: "What version introduced LCEL?"  ← follow-up
  - Return both answers as a list: [answer1, answer2]


HINT:
  from langchain.chains import create_history_aware_retriever
  from langchain.chains import create_retrieval_chain
  from langchain.chains.combine_documents import create_stuff_documents_chain
  from langchain_core.messages import HumanMessage, AIMessage


  contextualize_prompt — asks the LLM to rephrase the question
                         given history.
  qa_prompt           — answers based on context + history.
"""


import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

def conversational_rag(documents: list) -> list:
    """Returns [answer_turn1, answer_turn2] for a 2-turn RAG conversation."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docs = [Document(page_content=d) for d in documents]
    store = PGVector.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="rag_conversational",
        connection_string=os.environ["PG_CONNECTION_STRING"],
    )
    retriever = store.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    rewrite_prompt = ChatPromptTemplate.from_template(
        """
    Given the chat history and the follow-up question,
    rewrite the question so it is fully standalone.

    Chat History:
    {chat_history}

    Follow-up Question:
    {question}
    """
        )


    rewrite_chain = (
        rewrite_prompt
        | llm
        | StrOutputParser()
    )


    answer_prompt = ChatPromptTemplate.from_template(
        """Answer the question using ONLY the context below.

        Context:
        {context}

        Question:
        {question}
        """
            )


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    answer_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | answer_prompt
        | llm
        | StrOutputParser()
    )


    chat_history = []
    question_1 = "What is LangChain?"
    answer_1 = answer_chain.invoke(question_1)


    chat_history.extend([
        HumanMessage(content=question_1),
        AIMessage(content=answer_1),
    ])


    follow_up = "What version introduced LCEL?"
    standalone_question = rewrite_chain.invoke({
        "question": follow_up,
        "chat_history": chat_history,
    })


    answer_2 = answer_chain.invoke(standalone_question)


    return [answer_1, answer_2]
