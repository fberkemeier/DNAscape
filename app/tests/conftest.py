"""Small documented genomic examples for conversion regression tests."""

import pytest


@pytest.fixture
def sns_peak_payload() -> bytes:
    """Match the GEO SNS-seq BED9 column declarations, including IGV metadata."""

    return (
        b"# Column 1: chromosome\n"
        b"# Column 2: peak start\n"
        b"# Column 3: peak stop\n"
        b"# Column 4: average enrichment ratio\n"
        b"# Column 5: score for IGV\n"
        b"# Column 6: strand (not available in this file)\n"
        b"# Column 7: thick bar start in IGV\n"
        b"# Column 8: thick bar stop in IGV\n"
        b"# Column 9: red, green, and blue values for IGV\n"
        b"chr1\t100\t200\t10\t1000\tna\t100\t201\t8,119,38\n"
        b"chr1\t400\t800\t2\t200\tna\t400\t801\t8,119,38\n"
        b"chr1\t2200\t2300\t20\t1000\tna\t2200\t2301\t8,119,38\n"
    )
