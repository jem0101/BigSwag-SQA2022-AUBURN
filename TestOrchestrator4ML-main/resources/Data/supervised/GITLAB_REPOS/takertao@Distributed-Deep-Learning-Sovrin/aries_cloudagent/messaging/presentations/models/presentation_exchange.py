"""Handle presentation exchange information interface with non-secrets storage."""

from marshmallow import fields

from ...models.base_record import BaseRecord, BaseRecordSchema


class PresentationExchange(BaseRecord):
    """Represents a presentation exchange."""

    class Meta:
        """PresentationExchange metadata."""

        schema_class = "PresentationExchangeSchema"

    RECORD_TYPE = "presentation_exchange"
    RECORD_ID_NAME = "presentation_exchange_id"
    WEBHOOK_TOPIC = "presentations"
    LOG_STATE_FLAG = "debug.presentations"

    INITIATOR_SELF = "self"
    INITIATOR_EXTERNAL = "external"

    STATE_REQUEST_SENT = "request_sent"
    STATE_REQUEST_RECEIVED = "request_received"
    STATE_PRESENTATION_SENT = "presentation_sent"
    STATE_PRESENTATION_RECEIVED = "presentation_received"
    STATE_VERIFIED = "verified"

    def __init__(
        self,
        *,
        presentation_exchange_id: str = None,
        connection_id: str = None,
        thread_id: str = None,
        initiator: str = None,
        state: str = None,
        presentation_request: dict = None,
        presentation: dict = None,
        verified: str = None,
        error_msg: str = None,
        **kwargs
    ):
        """Initialize a new PresentationExchange."""
        super().__init__(presentation_exchange_id, state, **kwargs)
        self.connection_id = connection_id
        self.thread_id = thread_id
        self.initiator = initiator
        self.state = state
        self.presentation_request = presentation_request
        self.presentation = presentation
        self.verified = verified
        self.error_msg = error_msg

    @property
    def presentation_exchange_id(self) -> str:
        """Accessor for the ID associated with this exchange."""
        return self._id

    @property
    def record_value(self) -> dict:
        """Accessor for JSON record value generated for this presentation exchange."""
        result = self.tags
        for prop in ("presentation_request", "presentation", "error_msg"):
            val = getattr(self, prop)
            if val:
                result[prop] = val
        return result

    @property
    def record_tags(self) -> dict:
        """Accessor for the record tags generated for this presentation exchange."""
        result = {}
        for prop in ("connection_id", "thread_id", "initiator", "state", "verified"):
            val = getattr(self, prop)
            if val:
                result[prop] = val
        return result


class PresentationExchangeSchema(BaseRecordSchema):
    """Schema for serialization/deserialization of presentation exchange records."""

    class Meta:
        """PresentationExchangeSchema metadata."""

        model_class = PresentationExchange

    presentation_exchange_id = fields.Str(required=False)
    connection_id = fields.Str(required=False)
    thread_id = fields.Str(required=False)
    initiator = fields.Str(required=False)
    state = fields.Str(required=False)
    presentation_request = fields.Dict(required=False)
    presentation = fields.Dict(required=False)
    verified = fields.Str(required=False)
    error_msg = fields.Str(required=False)
