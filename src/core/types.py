from typing import Any

Message = dict[str, Any]
MessageList = list[Message]


class BaseModel:
    """
    Base class for language model.
    """

    # Invokes model with message_list
    def __call__(self, message_list: MessageList) -> str:
        raise NotImplementedError
