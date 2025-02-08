import os, json
import requests
from dotenv import load_dotenv
load_dotenv()

class Tools:
    def __init__(self) -> None:
        self.toolConfig = self._tools()

    """
    define the tool for agent to use
    use the google search here as the easiest example
    """
    def _tools(self):
        tools = [
            {
                'name_for_human': '谷歌搜索',
                'name_for_model': 'google_search',
                'description_for_model': '谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。',
                'parameters': [
                    {
                        'name': 'search_query',
                        'description': '搜索关键词或短语',
                        'required': True,
                        'schema': {'type': 'string'},
                    }
                ],
            }
        ]
        return tools

    """
    define how to use the tool
    Use the serper api here(https://serper.dev) serapi(https://serpapi.com) is also availlable
    """
    def google_search(self, search_query: str):
        url = "https://google.serper.dev/search"
        api_key = os.getenv('X-API-KEY')
        payload = json.dumps({"q": search_query})
        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload).json()
        print(response)
        try:
            return response['organic'][0]['snippet']
        except KeyError:
            return "No search results found."