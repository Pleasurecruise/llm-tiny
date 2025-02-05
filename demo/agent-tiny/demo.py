import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)

from agent_tiny.agent import Agent

agent = Agent('/root/autodl-tmp/llm-learning/model/Shanghai_AI_Laboratory/internlm2-chat-7b')

print(agent.system_prompt)

# response, _ = agent.text_completion(text='你好', history=[])
# print(response)

# response, _ = agent.text_completion(text='2025年的美国总统是谁？', history=[])
# print(response)

response, _ = agent.text_completion(text='Deepseek的 Janus Pro 是什么模型？', history=[])
print(response)