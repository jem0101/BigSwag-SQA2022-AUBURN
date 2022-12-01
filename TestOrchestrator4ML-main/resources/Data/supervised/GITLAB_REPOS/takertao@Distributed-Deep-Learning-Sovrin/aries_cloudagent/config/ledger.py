"""Ledger configuration."""

import asyncio
from collections import OrderedDict
import logging
import re
import sys

from aiohttp import ClientSession, ClientError, ClientResponse
import markdown
import prompt_toolkit
from prompt_toolkit.eventloop.defaults import use_asyncio_event_loop
from prompt_toolkit.formatted_text import HTML

from ..ledger.base import BaseLedger

from .base import ConfigError
from .injection_context import InjectionContext

LOGGER = logging.getLogger(__name__)


async def fetch_genesis_transactions(genesis_url: str):
    """Get genesis transactions."""
    headers = {}
    headers["Content-Type"] = "application/json"
    retry = 5
    LOGGER.info("Fetching genesis transactions from: %s", genesis_url)
    while True:
        try:
            async with ClientSession() as client_session:
                response: ClientResponse = await client_session.get(
                    genesis_url, headers=headers
                )
                if response.status < 200 or response.status >= 300:
                    raise ClientError("Bad response from server")
                genesis_txns = await response.text()
                return genesis_txns
        except ClientError as e:
            if retry:
                retry -= 1
                await asyncio.sleep(5.0)
            else:
                raise ConfigError("Error retrieving ledger genesis transactions") from e


async def ledger_config(
    context: InjectionContext, public_did: str, provision: bool = False
) -> bool:
    """Perform Indy ledger configuration."""

    # Fetch genesis transactions if necessary
    if not context.settings.get("ledger.genesis_transactions"):
        if context.settings.get("ledger.genesis_url"):
            context.settings[
                "ledger.genesis_transactions"
            ] = await fetch_genesis_transactions(context.settings["ledger.genesis_url"])
        elif context.settings.get("ledger.genesis_file"):
            try:
                genesis_path = context.settings["ledger.genesis_file"]
                LOGGER.info("Reading genesis transactions from: %s", genesis_path)
                with open(genesis_path, "r") as genesis_file:
                    context.settings["ledger.genesis_transactions"] = genesis_file.read(
                        -1
                    )
            except IOError as e:
                raise ConfigError("Error reading genesis transactions") from e

    ledger: BaseLedger = await context.inject(BaseLedger, required=False)
    if not ledger:
        LOGGER.info("Ledger instance not provided")
        return False
    elif ledger.LEDGER_TYPE != "indy":
        LOGGER.info("Non-indy ledger provided")
        return False

    async with ledger:
        # Check transaction author agreement acceptance
        taa_info = await ledger.get_txn_author_agreement()
        if taa_info["taa_required"] and public_did:
            taa_accepted = await ledger.get_latest_txn_author_acceptance()
            if (
                not taa_accepted
                or taa_info["taa_record"]["digest"] != taa_accepted["digest"]
            ):
                if not await accept_taa(ledger, taa_info, provision):
                    return False

        # Publish endpoint if necessary - skipped if TAA is required but not accepted
        endpoint = context.settings.get("default_endpoint")
        if public_did:
            await ledger.update_endpoint_for_did(public_did, endpoint)

    return True


async def accept_taa(ledger: BaseLedger, taa_info, provision: bool = False) -> bool:
    """Perform TAA acceptance."""

    if not sys.stdout.isatty():
        LOGGER.warning("Cannot accept TAA without interactive terminal")
        return False

    mechanisms = taa_info["aml_record"]["aml"]
    allow_opts = OrderedDict(
        [
            (
                "wallet_agreement",
                (
                    "Accept the transaction author agreement and store the "
                    + "acceptance in the wallet"
                ),
            ),
            (
                "on_file",
                (
                    "Acceptance of the transaction author agreement is on file "
                    + "in my organization"
                ),
            ),
        ]
    )
    if not provision:
        allow_opts["for_session"] = (
            "Accept the transaction author agreement for the duration of "
            + "the current session"
        )

    found = []
    for opt in allow_opts:
        if opt in mechanisms:
            found.append(opt)

    md = markdown.Markdown()
    taa_html = md.convert(taa_info["taa_record"]["text"])
    taa_html = re.sub(
        r"<h[1-6]>(.*?)</h[1-6]>", r"<p><strong>\1</strong></p>\n", taa_html
    )
    taa_html = re.sub(r"<li>(.*?)</li>", r" - \1", taa_html)

    taa_html = (
        "\n<strong>Transaction Author Agreement version "
        + taa_info["taa_record"]["version"]
        + "</strong>\n\n"
        + taa_html
    )

    # setup for prompt_toolkit
    use_asyncio_event_loop()

    prompt_toolkit.print_formatted_text(HTML(taa_html))

    opts = []
    num_mechanisms = {}
    for idx, opt in enumerate(found):
        num_mechanisms[str(idx + 1)] = opt
        opts.append(f" {idx+1}. {allow_opts[opt]}")
    opts.append(" X. Skip the transaction author agreement")
    opts_text = "\nPlease select an option:\n" + "\n".join(opts) + "\n[1]> "

    while True:
        try:
            opt = await prompt_toolkit.prompt(opts_text, async_=True)
        except EOFError:
            return False
        if not opt:
            opt = "1"
        opt = opt.strip()
        if opt in ("x", "X"):
            return False
        if opt in num_mechanisms:
            mechanism = num_mechanisms[opt]
            break

    await ledger.accept_txn_author_agreement(taa_info["taa_record"], mechanism)

    return True
