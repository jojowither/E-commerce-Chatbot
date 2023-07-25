import os
from dotenv import load_dotenv, dotenv_values 
from typing import List, Union, Tuple
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain import HuggingFaceHub


config = dotenv_values("../.env") 
os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY'] 
os.environ['HUGGINGFACEHUB_API_TOKEN'] = config['HUGGINGFACEHUB_API_TOKEN']
file_path = '../data/se_product.csv'
vectorstore_path = '../faiss_product/'
app = FastAPI()


# embeddings = HuggingFaceEmbeddings()
# llm = HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0, "max_length":512})

embeddings = OpenAIEmbeddings()

if not os.path.exists(vectorstore_path):
    loader = CSVLoader(file_path=file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()
    vectorstore = FAISS.from_documents(data, embeddings)
    vectorstore.save_local(vectorstore_path)
    print('Finsih save vectorstore')

else:
    vectorstore = FAISS.load_local(vectorstore_path, embeddings)

llm = ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo')
chain = ConversationalRetrievalChain.from_llm(llm=llm, 
                                              retriever=vectorstore.as_retriever())   


class ChatModel(BaseModel):
    question: str
    chat_history: Union[List[str], List[Tuple[str, str]]] = []


@app.post("/conversation")
async def conversation(data: ChatModel):
    response = chain({"question": data.question, "chat_history": data.chat_history})
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)