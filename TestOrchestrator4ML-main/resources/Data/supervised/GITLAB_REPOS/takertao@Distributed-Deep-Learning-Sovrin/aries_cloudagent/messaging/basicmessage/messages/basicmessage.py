"""Basic message."""

from datetime import datetime
from typing import Union

from marshmallow import fields

from ...agent_message import AgentMessage, AgentMessageSchema
from ...util import datetime_now, datetime_to_str
from ...valid import INDY_ISO8601_DATETIME

from ..message_types import BASIC_MESSAGE

HANDLER_CLASS = (
    "aries_cloudagent.messaging.basicmessage."
    + "handlers.basicmessage_handler.BasicMessageHandler"
)


class BasicMessage(AgentMessage):
    """Class defining the structure of a basic message."""

    class Meta:
        """Basic message metadata class."""

        handler_class = HANDLER_CLASS
        message_type = BASIC_MESSAGE
        schema_class = "BasicMessageSchema"

    def __init__(
        self,
        *,
        sent_time: Union[str, datetime] = None,
        content: str = None,
        localization: str = None,
        **kwargs
    ):
        """
        Initialize basic message object.

        Args:
            sent_time: Time message was sent
            content: message content
            localization: localization

        """
        super(BasicMessage, self).__init__(**kwargs)
        if not sent_time:
            sent_time = datetime_now()
        self.sent_time = datetime_to_str(sent_time)
        self.content = content
        self.localization = localization


class BasicMessageSchema(AgentMessageSchema):
    """Basic message schema class."""

    class Meta:
        """Basic message schema metadata."""

        model_class = BasicMessage

    localization = fields.Str(
        required=False,
        description="Localization",
        example="en-CA",
        data_key="l10n",
    )
    sent_time = fields.Str(
        required=False,
        description="Time message was sent, ISO8601 with space date/time separator",
        **INDY_ISO8601_DATETIME
    )
    content = fields.Str(
        required=True,
        description="Message content",
        example="Hello",
    )
