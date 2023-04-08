import openai
import tiktoken

# Type aliases
Messages = list[dict[str, str]]


class DSLChat:
    def __init__(self, model: str = "gpt-3.5-turbo-0301") -> None:
        self.messages: Messages = []
        self.model = model

    @classmethod
    def content2message(cls, role: str, content: str) -> dict[str, str]:
        return {
            "role": role,
            "content": content,
        }

    def completion(self) -> tuple[str, Messages]:
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages
        )
        response_content = completion.choices[0].message.content
        self.append_assistant(response_content)
        return response_content

    def __append_message(self, role: str, content: str) -> None:
        self.messages.append(self.content2message(role, content))

    def append_system(self, content: str) -> None:
        self.__append_message("system", content)

    def append_user(self, content: str) -> None:
        self.__append_message("user", content)

    def append_assistant(self, content: str) -> None:
        self.__append_message("assistant", content)

    def get_messages(self) -> Messages:
        return self.messages

    def num_tokens_from_messages(self) -> int:
        # ref: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        if self.model == "gpt-3.5-turbo-0301":
            # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_message = 4
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif self.model == "gpt-4-0314":
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {self.model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
        num_tokens = 0
        for message in self.messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens
