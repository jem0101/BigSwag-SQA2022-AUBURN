from asynctest import TestCase as AsyncTestCase

from aries_cloudagent.messaging.connections.messages.connection_request import ConnectionRequest
from aries_cloudagent.messaging.connections.models.connection_detail import ConnectionDetail
from aries_cloudagent.messaging.connections.models.diddoc import DIDDoc, PublicKey, PublicKeyType, Service
from aries_cloudagent.messaging.message_delivery import MessageDelivery
from aries_cloudagent.wallet.base import BaseWallet
from aries_cloudagent.wallet.basic import BasicWallet
from ....config.injection_context import InjectionContext
from ..models.connection_record import ConnectionRecord
from ....storage.base import BaseStorage
from ....storage.basic import BasicStorage

from ..manager import ConnectionManager, ConnectionManagerError


class TestConfig:

    test_seed = "testseed000000000000000000000001"
    test_did = "55GkHamhTU1ZbTbV2ab9DE"
    test_verkey = "3Dn1SJNPaCXcvvJvSbsFWP2xaCjMom3can8CQNhWrTRx"
    test_endpoint = "http://localhost"

    test_target_did = "GbuDUYXaUZRfHD2jeDuQuP"
    test_target_verkey = "9WCgWKUaAJj3VWxxtzvvMQN3AoFxoBtBDo9ntwJnVVCC"


    def make_did_doc(self, did, verkey):
        doc = DIDDoc(did=did)
        controller = did
        ident = "1"
        pk_value = verkey
        pk = PublicKey(
            did,
            ident,
            pk_value,
            PublicKeyType.ED25519_SIG_2018,
            controller,
            False,
        )
        doc.set(pk)
        recip_keys = [pk]
        router_keys = []
        service = Service(
            did,
            "indy",
            "IndyAgent",
            recip_keys,
            router_keys,
            self.test_endpoint,
        )
        doc.set(service)
        return doc


class TestConnectionManager(AsyncTestCase, TestConfig):
    def setUp(self):
        self.storage = BasicStorage()
        self.context = InjectionContext()
        self.context.injector.bind_instance(BaseStorage, self.storage)
        self.context.injector.bind_instance(BaseWallet, BasicWallet())
        self.manager = ConnectionManager(self.context)
        self.test_info = ConnectionRecord(
            my_did=self.test_did,
            their_did=self.test_target_did,
            their_role=None,
            state=ConnectionRecord.STATE_ACTIVE,
        )

    async def test_public_and_multi_use_fails(self):
        ci_awaitable = self.manager.create_invitation(public=True, multi_use=True)
        await self.assertAsyncRaises(ConnectionManagerError, ci_awaitable)

    async def test_non_multi_use_invitation_fails_on_reuse(self):
        connect_record, connect_invite = await self.manager.create_invitation(my_endpoint="testendpoint")

        delivery = MessageDelivery(recipient_verkey=connect_record.invitation_key)

        requestA = ConnectionRequest(
            connection=ConnectionDetail(did=self.test_target_did,
                                        did_doc=self.make_did_doc(self.test_target_did, self.test_target_verkey)),
            label="SameInviteRequestA",
        )

        await self.manager.receive_request(requestA, delivery)

        requestB = ConnectionRequest(
            connection=ConnectionDetail(did=self.test_did, did_doc=self.make_did_doc(self.test_did, self.test_verkey)),
            label="SameInviteRequestB",
        )

        # requestB fails because the invitation was not set to multi-use
        rr_awaitable = self.manager.receive_request(requestB, delivery)
        await self.assertAsyncRaises(ConnectionManagerError, rr_awaitable)

    async def test_multi_use_invitation(self):
        connect_record, connect_invite = await self.manager.create_invitation(my_endpoint="testendpoint", multi_use=True)

        delivery = MessageDelivery(recipient_verkey=connect_record.invitation_key)

        requestA = ConnectionRequest(
            connection=ConnectionDetail(did=self.test_target_did,
                                        did_doc=self.make_did_doc(self.test_target_did, self.test_target_verkey)),
            label="SameInviteRequestA",
        )

        await self.manager.receive_request(requestA, delivery)

        requestB = ConnectionRequest(
            connection=ConnectionDetail(did=self.test_did, did_doc=self.make_did_doc(self.test_did, self.test_verkey)),
            label="SameInviteRequestB",
        )

        await self.manager.receive_request(requestB, delivery)



