import pytest
from asynctest import mock as async_mock

from ....base_handler import HandlerException
from ....message_delivery import MessageDelivery
from ....request_context import RequestContext
from ....responder import MockResponder

from ...handlers import connection_request_handler as handler
from ...manager import ConnectionManagerError
from ...messages.connection_request import ConnectionRequest
from ...messages.problem_report import ProblemReport, ProblemReportReason


@pytest.fixture()
def request_context() -> RequestContext:
    ctx = RequestContext()
    ctx.message_delivery = MessageDelivery()
    yield ctx


class TestRequestHandler:
    @pytest.mark.asyncio
    @async_mock.patch.object(handler, "ConnectionManager")
    async def test_called(self, mock_conn_mgr, request_context):
        mock_conn_mgr.return_value.receive_request = async_mock.CoroutineMock()
        request_context.message = ConnectionRequest()
        handler_inst = handler.ConnectionRequestHandler()
        responder = MockResponder()
        await handler_inst.handle(request_context, responder)

        mock_conn_mgr.assert_called_once_with(request_context)
        mock_conn_mgr.return_value.receive_request.assert_called_once_with(
            request_context.message, request_context.message_delivery
        )
        assert not responder.messages

    @pytest.mark.asyncio
    @async_mock.patch.object(handler, "ConnectionManager")
    async def test_problem_report(self, mock_conn_mgr, request_context):
        mock_conn_mgr.return_value.receive_request = async_mock.CoroutineMock()
        mock_conn_mgr.return_value.receive_request.side_effect = ConnectionManagerError(
            error_code=ProblemReportReason.REQUEST_NOT_ACCEPTED
        )
        request_context.message = ConnectionRequest()
        handler_inst = handler.ConnectionRequestHandler()
        responder = MockResponder()
        await handler_inst.handle(request_context, responder)
        messages = responder.messages
        assert len(messages) == 1
        result, target = messages[0]
        assert (
            isinstance(result, ProblemReport)
            and result.problem_code == ProblemReportReason.REQUEST_NOT_ACCEPTED
        )
        assert target == {"target": None}
