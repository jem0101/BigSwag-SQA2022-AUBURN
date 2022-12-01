from ..query import Query
from ...message_types import QUERY

from unittest import mock, TestCase


class TestQuery(TestCase):
    test_query = "*"
    test_comment = "comment"

    def test_init(self):
        query = Query(query=self.test_query, comment=self.test_comment)
        assert query.query == self.test_query
        assert query.comment == self.test_comment

    def test_type(self):
        query = Query(query=self.test_query, comment=self.test_comment)
        assert query._type == QUERY

    @mock.patch("aries_cloudagent.messaging.discovery.messages.query.QuerySchema.load")
    def test_deserialize(self, mock_query_schema_load):
        obj = {"obj": "obj"}

        query = Query.deserialize(obj)
        mock_query_schema_load.assert_called_once_with(obj)

        assert query is mock_query_schema_load.return_value

    @mock.patch("aries_cloudagent.messaging.discovery.messages.query.QuerySchema.dump")
    def test_serialize(self, mock_query_schema_dump):
        query = Query(query=self.test_query, comment=self.test_comment)

        query_dict = query.serialize()
        mock_query_schema_dump.assert_called_once_with(query)

        assert query_dict is mock_query_schema_dump.return_value


class TestQuerySchema(TestCase):

    query = Query(query="*", comment="comment")

    def test_make_model(self):
        data = self.query.serialize()
        model_instance = Query.deserialize(data)
        assert isinstance(model_instance, Query)
