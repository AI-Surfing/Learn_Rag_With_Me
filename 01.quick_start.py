import torch
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


#### INDEXING ####
# ----------------- 配置项 ---------------------------- #
data_file = "../data/中华人民共和国证券法(2019修订).pdf"
model_path = "/data/models/Baichuan2-13B-Chat"
embed_path = "/data/models/bge-large-zh-v1.5"
# ----------------- 加载embedding模型 ----------------- #
embeddings = HuggingFaceEmbeddings(
    model_name=embed_path,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)
# ----------------- 加载LLM -------------------------- #
tokenizer = AutoTokenizer.from_pretrained(model_path,
                                          device_map="auto",
                                          trust_remote_code=True,
                                          torch_dtype=torch.float16)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="auto",
)

pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
)

llm = HuggingFacePipeline(pipeline=pipeline)
# ----------------- 加载文件 ----------------- #
loader = PyPDFLoader(data_file)
documents = loader.load_and_split()
text_splitter = RecursiveCharacterTextSplitter(separators=["。"], chunk_size=512, chunk_overlap=32)
texts_chunks = text_splitter.split_documents(documents)
# ----------------- 存入向量库 ---------------- #
vectorstore = Chroma.from_documents(texts_chunks, embeddings)
retriever = vectorstore.as_retriever()
#### RETRIEVAL and GENERATION ####

# Prompt
template = '''You are an assistant for question-answering tasks. Use the following pieces of retrieved context to 
answer the question. If you don’t know the answer, just say that you don’t know. Use three sentences maximum and 
keep the answer concise.
Question: {question}
Context: {context}
Answer:""
'''
prompt = PromptTemplate.from_template(template)


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
answer = rag_chain.invoke("上市公司年报披露有哪些规定?")
print(answer)
