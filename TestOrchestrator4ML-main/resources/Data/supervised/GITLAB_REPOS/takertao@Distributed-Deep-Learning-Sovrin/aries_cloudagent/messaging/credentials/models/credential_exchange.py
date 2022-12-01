"""Handle credential exchange information interface with non-secrets storage."""

from marshmallow import fields

from ...models.base_record import BaseRecord, BaseRecordSchema


class CredentialExchange(BaseRecord):
    """Represents a credential exchange."""

    class Meta:
        """CredentialExchange metadata."""

        schema_class = "CredentialExchangeSchema"

    RECORD_TYPE = "credential_exchange"
    RECORD_ID_NAME = "credential_exchange_id"
    WEBHOOK_TOPIC = "credentials"
    LOG_STATE_FLAG = "debug.credentials"

    INITIATOR_SELF = "self"
    INITIATOR_EXTERNAL = "external"

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
        credential_offer: dict = None,
        credential_request: dict = None,
        credential_request_metadata: dict = None,
        credential_id: str = None,
        raw_credential: dict = None,
        credential: dict = None,
        credential_values: dict = None,
        auto_issue: bool = False,
        error_msg: str = None,
        **kwargs,
    ):
        """Initialize a new CredentialExchange."""
        super().__init__(credential_exchange_id, state, **kwargs)
        self.connection_id = connection_id
        self.thread_id = thread_id
        self.parent_thread_id = parent_thread_id
        self.initiator = initiator
        self.state = state
        self.credential_definition_id = credential_definition_id
        self.schema_id = schema_id
        self.credential_offer = credential_offer
        self.credential_request = credential_request
        self.credential_request_metadata = credential_request_metadata
        self.credential_id = credential_id
        self.credential = credential
        self.raw_credential = raw_credential
        self.credential_values = credential_values
        self.auto_issue = auto_issue
        self.error_msg = error_msg

    @property
    def credential_exchange_id(self) -> str:
        """Accessor for the ID associated with this exchange."""
        return self._id

    @property
    def record_value(self) -> dict:
        """Accessor to for the JSON record value props for this credential exchange."""
        return {
            prop: getattr(self, prop)
            for prop in (
                "credential_offer",
                "credential_request",
                "credential_request_metadata",
                "error_msg",
                "auto_issue",
                "credential_values",
                "credential",
                "raw_credential",
                "parent_thread_id",
            )
        }

    @property
    def record_tags(self) -> dict:
        """Accessor for the record tags generated for this credential exchange."""
        return {
            prop: getattr(self, prop)
            for prop in (
                "connection_id",
                "thread_id",
                "initiator",
                "state",
                "credential_definition_id",
                "schema_id",
                "credential_id",
            )
        }


class CredentialExchangeSchema(BaseRecordSchema):
    """Schema to allow serialization/deserialization of credential exchange records."""

    class Meta:
        """CredentialExchangeSchema metadata."""

        model_class = CredentialExchange

    credential_exchange_id = fields.Str(required=False)
    connection_id = fields.Str(required=False)
    thread_id = fields.Str(required=False)
    parent_thread_id = fields.Str(required=False)
    initiator = fields.Str(required=False)
    state = fields.Str(required=False)
    credential_definition_id = fields.Str(required=False)
    schema_id = fields.Str(required=False)
    credential_offer = fields.Dict(required=False)
    credential_request = fields.Dict(required=False)
    credential_request_metadata = fields.Dict(required=False)
    credential_id = fields.Str(required=False)
    credential = fields.Dict(required=False)
    raw_credential = fields.Dict(required=False)
    auto_issue = fields.Bool(required=False)
    credential_values = fields.Dict(required=False)
    error_msg = fields.Str(required=False)
