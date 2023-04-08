# %% imports
from dotenv import load_dotenv
from utils.openai import chat_completion
from utils.openai import num_tokens_from_messages

# %% loadconfig
load_dotenv()

# %% main
messages = [
    {"role": "system", "content": "日本語で回答します。"},
    {"role": "assistant", "content": "日本語で回答します。"},
]

chat_completion(messages)
num: int = num_tokens_from_messages(messages)
print(num)
# %%
