"""Message type identifiers for Feature Discovery."""

MESSAGE_FAMILY = "did:sov:BzCbsNYhMrjHiqZDTUASHg;spec/discover-features/1.0"

DISCLOSE = f"{MESSAGE_FAMILY}/disclose"
QUERY = f"{MESSAGE_FAMILY}/query"

MESSAGE_PACKAGE = "aries_cloudagent.messaging.discovery.messages"

MESSAGE_TYPES = {
    DISCLOSE: f"{MESSAGE_PACKAGE}.disclose.Disclose",
    QUERY: f"{MESSAGE_PACKAGE}.query.Query",
}
