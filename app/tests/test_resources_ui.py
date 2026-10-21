"""Resources links and citation actions use the displayed publication."""

from __future__ import annotations

import asyncio

import pytest
from nicegui.testing import user_simulation

from app.icons import GITHUB_ICON_SVG
from app.main import index
from app.pages import resources


def test_resources_links_and_citation_copy_and_download(monkeypatch: pytest.MonkeyPatch) -> None:
    copied: list[str] = []
    monkeypatch.setattr(resources.ui.clipboard, "write", copied.append)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-resources").click()
            await user.should_see("Project links and citation guidance")
            await user.should_see("Installation notes, tutorials, and Python API guidance.")
            await user.should_see("Francisco Berkemeier and Michael A. Boemo (2026).")
            await user.should_see(resources.CITATION_TITLE)
            citation = next(iter(user.find(marker="resource-citation-text").elements))
            assert citation.text == resources.CITATION_TEXT
            assert "dn-resource-citation" in citation.parent_slot.parent._classes
            await user.should_not_see("Formal citation metadata can be added")

            documentation = next(iter(user.find(marker="resource-documentation").elements))
            repository = next(iter(user.find(marker="resource-repository").elements))
            assert documentation._props["href"] == resources.DOCS_URL
            assert repository._props["href"] == resources.GITHUB_URL
            assert documentation._props["target"] == repository._props["target"] == "_blank"
            assert [child.text for child in documentation.default_slot.children if hasattr(child, "text")] == [
                "DNAscape Read the Docs",
            ]
            assert repository.default_slot.children[0].content == GITHUB_ICON_SVG
            assert repository.default_slot.children[1].text == "DNAscape on GitHub"
            cards = documentation.parent_slot.parent.parent_slot.children
            assert cards[0] is documentation.parent_slot.parent
            assert cards[1] is repository.parent_slot.parent
            assert "dn-resource-citation-panel" in cards[2]._classes

            user.find(marker="action-copy-citation").click()
            assert copied == [resources.CITATION_TEXT]
            user.find(marker="action-download-bibtex").click()
            download = await user.download.next()
            bibtex = download.content.decode("utf-8")
            assert bibtex == resources.CITATION_BIBTEX
            assert "author = {Berkemeier, Francisco and Boemo, Michael A.}" in bibtex
            assert "title = {{DNAscape}: Software for mapping and simulating DNA replication kinetics}" in bibtex
            assert "journal = {bioRxiv}" in bibtex
            assert "year = {2026}" in bibtex
            assert "doi" not in bibtex.casefold()

    asyncio.run(scenario())
