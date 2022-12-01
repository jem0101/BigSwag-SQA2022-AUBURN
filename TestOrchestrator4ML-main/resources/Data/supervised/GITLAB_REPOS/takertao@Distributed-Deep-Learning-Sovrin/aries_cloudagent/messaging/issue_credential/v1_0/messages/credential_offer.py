"""A credential offer content message."""


from typing import Sequence

from marshmallow import fields

from ....agent_message import AgentMessage, AgentMessageSchema
from ....decorators.attach_decorator import AttachDecorator, AttachDecoratorSchema
from ..message_types import CREDENTIAL_OFFER
from .inner.credential_preview import CredentialPreview, CredentialPreviewSchema


HANDLER_CLASS = (
    "aries_cloudagent.messaging.issue_credential.v1_0.handlers."
    + "credential_offer_handler.CredentialOfferHandler"
)


class CredentialOffer(AgentMessage):
    """Class representing a credential offer."""

    class Meta:
        """CredentialOffer metadata."""

        handler_class = HANDLER_CLASS
        schema_class = "CredentialOfferSchema"
        message_type = CREDENTIAL_OFFER

    def __init__(
        self,
        _id: str = None,
        *,
        comment: str = None,
        credential_preview: CredentialPreview = None,
        offers_attach: Sequence[AttachDecorator] = None,
        **kwargs
    ):
        """
        Initialize credential offer object.

        Args:
            comment: optional human-readable comment
            credential_preview: credential preview
            offers_attach: list of offer attachments

        """
        super().__init__(_id=_id, **kwargs)
        self.comment = comment
        self.credential_preview = (
            credential_preview if credential_preview else CredentialPreview()
        )
        self.offers_attach = list(offers_attach) if offers_attach else []

    def indy_offer(self, index: int = 0):
        """
        Retrieve and decode indy offer from attachment.

        Args:
            index: ordinal in attachment list to decode and return
                (typically, list has length 1)

        """
        return self.offers_attach[index].indy_dict


class CredentialOfferSchema(AgentMessageSchema):
    """Credential offer schema."""

    class Meta:
        """Credential offer schema metadata."""

        model_class = CredentialOffer

    comment = fields.Str(required=False, allow_none=False)
    credential_preview = fields.Nested(CredentialPreviewSchema, required=False)
    offers_attach = fields.Nested(
        AttachDecoratorSchema,
        required=True,
        many=True,
        data_key="offers~attach"
    )
