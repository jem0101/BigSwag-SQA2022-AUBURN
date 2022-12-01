import pytest

from ....base_handler import HandlerException
from ....message_delivery import MessageDelivery
from ....request_context import RequestContext
from ....responder import MockResponder

from ...handlers.connection_invitation_handler import ConnectionInvitationHandler
from ...messages.connection_invitation import ConnectionInvitation
from ...messages.problem_report import ProblemReport, ProblemReportReason


@pytest.fixture()
def request_context() -> RequestContext:
    ctx = RequestContext()
    ctx.message_delivery = MessageDelivery()
    yield ctx


class TestInvitationHandler:
    @pytest.mark.asyncio
    async def test_problem_report(self, request_context):
        request_context.message = ConnectionInvitation()
        handler = ConnectionInvitationHandler()
        responder = MockResponder()
        await handler.handle(request_context, responder)
        messages = responder.messages
        assert len(messages) == 1
        result, target = messages[0]
        assert (
            isinstance(result, ProblemReport)
            and result.problem_code == ProblemReportReason.INVITATION_NOT_ACCEPTED
        )
        assert not target
