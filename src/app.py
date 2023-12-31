import os
import yaml
import re
from dotenv import load_dotenv, dotenv_values 
from typing import List, Union, Tuple
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain import HuggingFaceHub


env_config = dotenv_values("../.env") 
USE_OPENAI_LLM = yaml.safe_load(env_config['USE_OPENAI_LLM'])
config = yaml.safe_load(open("config.yaml"))
file_path = config['file_path']
vectorstore_name = config['vectorstore_name']
product_df = pd.read_csv(file_path)
app = FastAPI()

def get_vectorstore_path(vectorstore_path, file_path, embeddings):
    if not os.path.exists(vectorstore_path):
        loader = CSVLoader(file_path=file_path, encoding="utf-8", csv_args={'delimiter': ','})
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
        documents = text_splitter.split_documents(loader.load())
        
        for idx, document in enumerate(documents):
            try:
                sale_num_content = document.page_content.split('\n')[1]
            except IndexError:
                sale_num_content = documents[idx-1].metadata['sale_no']

            match = re.search(r'\d+', sale_num_content)
            if match:
                documents[idx].metadata['sale_no'] = match.group()
            else:
                documents[idx].metadata['sale_no'] = '0000'
        vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=vectorstore_path)
        print('Finsih save vectorstore')
    else:
        vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)

    return vectorstore


if USE_OPENAI_LLM:
    os.environ['OPENAI_API_KEY'] = env_config['OPENAI_API_KEY'] 
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo')

    vectorstore_path = f'../{vectorstore_name}_openai/'
    vectorstore = get_vectorstore_path(vectorstore_path, file_path, embeddings)
else:
    # recommend to use GPU
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = env_config['HUGGINGFACEHUB_API_TOKEN']
    embeddings = HuggingFaceEmbeddings()
    llm = HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0, "max_length":512})

    vectorstore_path = f'../{vectorstore_name}_hf/'
    vectorstore = get_vectorstore_path(vectorstore_path, file_path, embeddings)


chain = ConversationalRetrievalChain.from_llm(llm=llm, 
                                              retriever=vectorstore.as_retriever(),
                                              return_source_documents=True)   


def get_metadata(documents):
    sale_no_list = []
    for document in documents:
        sale_no_list.append(int(document.metadata['sale_no']))
    # product_df[product_df["銷售編號"].isin(sale_no_list)]
    return sale_no_list


class ChatModel(BaseModel):
    question: str
    chat_history: Union[List[str], List[Tuple[str, str]]] = []


@app.post("/conversation")
async def conversation(data: ChatModel):
    response = chain({"question": data.question, "chat_history": data.chat_history})
    # can use this metadata to get the product from db
    sale_no_list = get_metadata(response["source_documents"])
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config['API_PORT'])