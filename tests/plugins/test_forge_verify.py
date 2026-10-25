from __future__ import annotations

import pytest
from fake_llama_tools import STATS

from fle_studio_forge import verify

SAMPLE = STATS.format(kld=0.012345)


def test_the_statistics_block_is_parsed():
    s = verify.parse("perplexity: 40960 tokens\n" + SAMPLE)
    assert s["complete"] and s["kld_mean"] == pytest.approx(0.012345) and s["kld_mean_se"] == pytest.approx(0.000123)
    assert s["kld_99"] == pytest.approx(0.1) and s["kld_max"] == pytest.approx(1.234567)
    assert s["kld_median"] == pytest.approx(0.005) and s["kld_95"] == pytest.approx(0.05)
    assert s["ppl_ratio"] == pytest.approx(1.021988) and s["ppl_q"] == pytest.approx(6.234567)
    assert s["ppl_base"] == pytest.approx(6.1) and s["ppl_correlation"] == pytest.approx(99.12)
    assert s["top1"] == pytest.approx(95.123) and s["top1_se"] == pytest.approx(0.05)
    assert s["delta_p_mean"] == pytest.approx(-0.123) and s["delta_p_rms"] == pytest.approx(3.21)
    assert s["tokens"] == 40960


@pytest.mark.parametrize("pm", ["±", "+/-", "Â±", "�", "��"])
def test_the_plus_minus_sign_may_arrive_mangled(pm):
    s = verify.parse(SAMPLE.replace("±", pm))
    assert s["kld_mean"] == pytest.approx(0.012345) and s["kld_mean_se"] == pytest.approx(0.000123)
    assert s["top1_se"] == pytest.approx(0.05)


def test_a_value_without_an_error_estimate_and_nan_are_tolerated():
    s = verify.parse("Mean    KLD:   0.5\nMaximum KLD: nan\nSame top p: 90.0 %\n")
    assert s["kld_mean"] == 0.5 and "kld_mean_se" not in s
    assert s["kld_max"] is None and s["top1"] == 90.0 and s["complete"]


def test_no_statistics_means_incomplete():
    s = verify.parse("llama_model_load: error loading model\n")
    assert not s["complete"] and "kld_mean" not in s


def test_the_card_turns_percent_into_fractions_and_computes_bits_per_weight():
    s = verify.parse(SAMPLE)
    c = verify.card(
        s,
        name="q",
        size=1_000_000,
        params=2_000_000,
        reference="bf16.gguf",
        text="wiki.txt",
        context=2048,
        chunks=20,
        tool_version="9999",
    )
    assert c["complete"] and c["bpw"] == 4.0 and c["top1"] == pytest.approx(0.95123)
    assert c["top1_se"] == pytest.approx(0.0005) and c["kld"] == pytest.approx(0.012345)
    assert c["reference"] == "bf16.gguf" and "same text" in c["note"]
    empty = verify.card({}, name="q", size=1, params=None, reference="", text="", context=1, chunks=1, tool_version="")
    assert not empty["complete"] and empty["bpw"] is None and empty["top1"] is None


def test_reference_cards_come_from_the_catalog():
    cards = verify.reference_cards()
    assert cards and all(len(c["kld"]) == 3 and c["size"] for c in cards)
