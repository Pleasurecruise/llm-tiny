import os
from copy import copy
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

os.environ['CURL_CA_BUNDLE'] = ''
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

"""
Base class for embeddings
"""
class BaseEmbeddings:
    """
    init the model path or api ending
    """
    def __init__(self, path: str, is_api: bool) -> None:
        self.path = path
        self.is_api = is_api
    """
    text: the text need to change to vector
    model: the model to use
    return: embedding vector
    """
    def get_embedding(self, text: str, model: str) -> List[float]:
        raise NotImplementedError
    """
    calculate cosine similarity between two vectors
    """
    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude

"""
class for OpenAI embeddings
close source, use through api key
document url:
"""
class OpenAIEmbedding(BaseEmbeddings):
    """
    inherit the base embedding
    """
    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            from openai import OpenAI
            self.client = OpenAI()
            self.client.api_key = os.getenv("OPENAI_API_KEY")
            self.client.base_url = os.getenv("OPENAI_BASE_URL")
    
    def get_embedding(self, text: str, model: str = "text-embedding-3-large") -> List[float]:
        if self.is_api:
            text = text.replace("\n", " ")
            return self.client.embeddings.create(input=[text], model=model).data[0].embedding
        else:
            raise NotImplementedError

"""
class for Jina embeddings
"""
class JinaEmbedding(BaseEmbeddings):

    def __init__(self, path: str = 'jinaai/jina-embeddings-v2-base-zh', is_api: bool = False) -> None:
        super().__init__(path, is_api)
        self._model = self.load_model()
        
    def get_embedding(self, text: str) -> List[float]:
        return self._model.encode([text])[0].tolist()
    
    def load_model(self):
        import torch
        from transformers import AutoModel
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model = AutoModel.from_pretrained(self.path, trust_remote_code=True).to(device)
        return model

"""
class for Zhipu embeddings
"""
class ZhipuEmbedding(BaseEmbeddings):

    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            from zhipuai import ZhipuAI
            self.client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY")) 
    
    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
        model="embedding-3",
        input=text,
        )
        return response.data[0].embedding

"""
class for Dashscope embeddings
"""
class DashscopeEmbedding(BaseEmbeddings):

    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            import dashscope
            dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
            self.client = dashscope.TextEmbedding

    def get_embedding(self, text: str, model: str='text-embedding-v1') -> List[float]:
        response = self.client.call(
            model=model,
            input=text
        )
        return response.output['embeddings'][0]['embedding']