#TASK 17: RAG Agent with Retriever as Tool
"""
-------------------------------------------
Convert the vector store retriever into a LangChain Tool,
then wrap it in a ReAct agent.  This lets the agent DECIDE
when to retrieve rather than always retrieving.


Steps:
  1. Build a PGVector store from RAG_DOCUMENTS.
  2. Wrap the retriever in a Tool named "knowledge_base".
  3. Create a ReAct agent with that tool.
  4. Ask: "What distance metrics does pgvector support?"
  5. Return the final answer string.


HINT:
  from langchain.tools.retriever import create_retriever_tool
  retriever_tool = create_retriever_tool(
      retriever,
      name="knowledge_base",
      description="Search the knowledge base for technical info."
  )
  Then pass [retriever_tool] to create_react_agent.
"""
import os
from typing import List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_community.vectorstores.pgvector import PGVector


def rag_agent(question: str) -> str:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docs = [Document(page_content=d) for d in RAG_DOCUMENTS]


    store = PGVector.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="rag_agent",
        connection_string=os.environ["PG_CONNECTION_STRING"],
        pre_delete_collection=True,
    )

    retriever = store.as_retriever()

    @tool
    def knowledge_base(query: str) -> str:
        """Search the knowledge base for technical information."""
        docs = retriever.invoke(query)
        return "\n\n".join(d.page_content for d in docs)


    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


    agent = create_agent(
        llm,
        tools=[knowledge_base],
    )


    result = agent.invoke({
        "messages": [
            {"role": "user", "content": question}
        ]
    })
    if "output" in result:
        return result["output"]


    if "messages" in result and len(result["messages"]) > 0:
        last_message = result["messages"][-1]
        return (
        last_message["content"]
        if isinstance(last_message, dict)
        else last_message.content
        )


    raise ValueError("Agent did not return a valid response")

print("\n[Task 17] RAG Agent")
agent_ans = rag_agent("What distance metrics does pgvector support?")
print(" ", agent_ans)