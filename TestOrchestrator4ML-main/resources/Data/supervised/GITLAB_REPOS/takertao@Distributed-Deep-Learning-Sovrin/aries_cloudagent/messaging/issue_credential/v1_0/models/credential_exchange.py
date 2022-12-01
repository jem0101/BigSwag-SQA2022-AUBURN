"""Aries#0036 v1.0 credential exchange information with non-secrets storage."""

from marshmallow import fields
from marshmallow.validate import OneOf

from ....models.base_record import BaseRecord, BaseRecordSchema
from ....valid import INDY_CRED_DEF_ID, INDY_SCHEMA_ID, UUIDFour


class V10CredentialExchange(BaseRecord):
    """Represents an Aries#0036 credential exchange."""

    class Meta:
        """CredentialExchange metadata."""

        schema_class = "V10CredentialExchangeSchema"

    RECORD_TYPE = "credential_exchange_v10"
    RECORD_ID_NAME = "credential_exchange_id"
    WEBHOOK_TOPIC = "issue_credential"

    INITIATOR_SELF = "self"
    INITIATOR_EXTERNAL = "external"

    STATE_PROPOSAL_SENT = "proposal_sent"
    STATE_PROPOSAL_RECEIVED = "proposal_received"
    STATE_OFFER_SENT = "offer_sent"
    STATE_OFFER_RECEIVED = "offer_received"
    STATE_REQUEST_SENT = "request_sent"
    STATE_REQUEST_RECEIVED = "request_received"
    STATE_ISSUED = "issued"
    STATE_CREDENTIAL_RECEIVED = "credential_received"
    STATE_STORED = "stored"

    def __init__(
        self,
        *,
        credential_exchange_id: str = None,
        connection_id: str = None,
        thread_id: str = None,
        parent_thread_id: str = None,
        initiator: str = None,
        state: str = None,
        credential_definition_id: str = None,
        schema_id: str = None,
        credential_proposal_dict: dict = None,  # serialized credential proposal message
        credential_offer: dict = None,  # indy credential offer
        credential_request: dict = None,  # indy credential request
        credential_request_metadata: dict = None,
        credential_id: str = None,
        raw_credential: dict = None,  # indy credential as received
        credential: dict = None,  # indy credential as stored
        auto_offer: bool = False,
        auto_issue: bool = False,
        error_msg: str = None,
        **kwargs
    ):
        """Initialize a new V10CredentialExchange."""
        super().__init__(credential_exchange_id, state, **kwargs)
        self._id = credential_exchange_id
        self.connection_id = connection_id
        self.thread_id = thread_id
        self.parent_thread_id = parent_thread_id
        self.initiator = initiator
        self.state = state
        self.credential_definition_id = credential_definition_id
        self.schema_id = schema_id
        self.credential_proposal_dict = credential_proposal_dict
        self.credential_offer = credential_offer
        self.credential_request = credential_request
        self.credential_request_metadata = credential_request_metadata
        self.credential_id = credential_id
        self.raw_credential = raw_credential
        self.credential = credential
        self.auto_offer = auto_offer
        self.auto_issue = auto_issue
        self.error_msg = error_msg

    @property
    def credential_exchange_id(self) -> str:
        """Accessor for the ID associated with this exchange."""
        return self._id

    @property
    def record_value(self) -> dict:
        """Accessor for the JSON record value generated for this credential exchange."""
        result = self.tags
        for prop in (
            "credential_proposal_dict",
            "credential_offer",
            "credential_request",
            "credential_request_metadata",
            "error_msg",
            "auto_offer",
            "auto_issue",
            "raw_credential",
            "credential",
            "parent_thread_id",
        ):
            val = getattr(self, prop)
            if val:
                result[prop] = val
        return result

    @property
    def record_tags(self) -> dict:
        """Accessor for the record tags generated for this credential exchange."""
        result = {}
        for prop in (
            "connection_id",
            "thread_id",
            "initiator",
            "state",
            "credential_definition_id",
            "schema_id",
            "credential_id",
        ):
            val = getattr(self, prop)
            if val:
                result[prop] = val
        return result


class V10CredentialExchangeSchema(BaseRecordSchema):
    """Schema to allow serialization/deserialization of credential exchange records."""

    class Meta:
        """V10CredentialExchangeSchema metadata."""

        model_class = V10CredentialExchange

    credential_exchange_id = fields.Str(
        required=False,
        description="Credential exchange identifier",
        example=UUIDFour.EXAMPLE,
    )
    connection_id = fields.Str(
        required=False,
        description="Connection identifier",
        example=UUIDFour.EXAMPLE,
    )
    thread_id = fields.Str(
        required=False,
        description="Thread identifier",
        example=UUIDFour.EXAMPLE,
    )
    parent_thread_id = fields.Str(
        required=False,
        description="Parent thread identifier",
        example=UUIDFour.EXAMPLE,
    )
    initiator = fields.Str(
        required=False,
        description="Issue-credential exchange initiator: self or external",
        example=V10CredentialExchange.INITIATOR_SELF,
        validate=OneOf(["self", "external"]),
    )
    state = fields.Str(
        required=False,
        description="Issue-credential exchange state",
        example=V10CredentialExchange.STATE_STORED,
    )
    credential_definition_id = fields.Str(
        required=False,
        description="Credential definition identifier",
        **INDY_CRED_DEF_ID
    )
    schema_id = fields.Str(
        required=False,
        description="Schema identifier",
        **INDY_SCHEMA_ID
    )
    credential_proposal_dict = fields.Dict(
        required=False,
        description="Serialized credential proposal message"
    )
    credential_offer = fields.Dict(
        required=False,
        description="(Indy) credential offer",
    )
    credential_request = fields.Dict(
        required=False,
        description="(Indy) credential request",
    )
    credential_request_metadata = fields.Dict(
        required=False,
        description="(Indy) credential request metadata",
    )
    credential_id = fields.Str(
        required=False,
        description="Credential identifier",
        example=UUIDFour.EXAMPLE,
    )
    raw_credential = fields.Dict(
        required=False,
        description="Credential as received, prior to storage in holder wallet"
    )
    credential = fields.Dict(
        required=False,
        description="Credential as stored",
    )
    auto_offer = fields.Bool(
        required=False,
        description="Holder choice to accept offer in this credential exchange",
        example=False,
    )
    auto_issue = fields.Bool(
        required=False,
        description="Issuer choice to issue to request in this credential exchange",
        example=False,
    )
    error_msg = fields.Str(
        required=False,
        description="Error message",
        example="credential definition identifier is not set in proposal",
    )
