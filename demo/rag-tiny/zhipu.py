import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)

from rag_tiny.vectorbase import VectorStore
from rag_tiny.utils import ReadFiles
from rag_tiny.llm import OpenAIChat, InternLMChat
from rag_tiny.base import JinaEmbedding, ZhipuEmbedding

"""
Use the utils 'ReadFiles.getcontent' in rag-tiny to process the data in 'data'(cut and process)
"""
docs = ReadFiles('../../data').get_content(max_token_len=600, cover_content=150)
"""
load the vectorbase 'VectorStore' class
"""
vector = VectorStore(docs)
# """
# Load the json from the local storage
# """
# vector.load_vector('../../storage') 
"""
Create the embedding model through the base model
Using Zhipu here
"""
embedding = ZhipuEmbedding()
"""
Start to transform here
"""
vector.get_vector(EmbeddingModel=embedding)
"""
Store the processed text in 'storage' dictionary for next time to directly use
"""
vector.persist(path='../../storage')

question = '什么是attention机制？'
content = vector.query(question, EmbeddingModel=ZhipuEmbedding(), k=1)
chat = InternLMChat(path='/root/autodl-tmp/llm-learning/model/Shanghai_AI_Laboratory/internlm2-chat-7b')
print(chat.chat(question, [], content))