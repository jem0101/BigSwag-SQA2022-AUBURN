"""A credential request content message."""


from typing import Sequence

from marshmallow import fields

from ....agent_message import AgentMessage, AgentMessageSchema
from ....decorators.attach_decorator import AttachDecorator, AttachDecoratorSchema
from ..message_types import CREDENTIAL_REQUEST


HANDLER_CLASS = (
    "aries_cloudagent.messaging.issue_credential.v1_0.handlers."
    + "credential_request_handler.CredentialRequestHandler"
)


class CredentialRequest(AgentMessage):
    """Class representing a credential request."""

    class Meta:
        """CredentialRequest metadata."""

        handler_class = HANDLER_CLASS
        schema_class = "CredentialRequestSchema"
        message_type = CREDENTIAL_REQUEST

    def __init__(
        self,
        _id: str = None,
        *,
        comment: str = None,
        requests_attach: Sequence[AttachDecorator] = None,
        **kwargs
    ):
        """
        Initialize credential request object.

        Args:
            requests_attach: requests attachments
            comment: optional comment

        """
        super().__init__(_id=_id, **kwargs)
        self.comment = comment
        self.requests_attach = list(requests_attach) if requests_attach else []

    def indy_cred_req(self, index: int = 0):
        """
        Retrieve and decode indy credential request from attachment.

        Args:
            index: ordinal in attachment list to decode and return
                (typically, list has length 1)

        """
        return self.requests_attach[index].indy_dict


class CredentialRequestSchema(AgentMessageSchema):
    """Credential request schema."""

    class Meta:
        """Credential request schema metadata."""

        model_class = CredentialRequest

    comment = fields.Str(required=False)
    requests_attach = fields.Nested(
        AttachDecoratorSchema,
        required=True,
        many=True,
        data_key="requests~attach"
    )
