"""Keyed digest behaviour."""

from __future__ import annotations

import pytest

from gateway.telemetry.hashing import DIGEST_LENGTH, HashKeyError, keyed_digest


def test_digest_is_deterministic():
    assert keyed_digest("prompt", key="k") == keyed_digest("prompt", key="k")


def test_digest_length_fits_the_hash_columns():
    assert len(keyed_digest("anything", key="k")) == DIGEST_LENGTH


def test_different_content_yields_different_digests():
    assert keyed_digest("a", key="k") != keyed_digest("b", key="k")


def test_different_keys_yield_different_digests():
    """Keying is what stops a short prompt being brute-forced from its hash."""
    assert keyed_digest("prompt", key="k1") != keyed_digest("prompt", key="k2")


def test_digest_does_not_contain_the_content():
    secret = "patient name is Alex Doe"
    assert secret not in keyed_digest(secret, key="k")


def test_bytes_and_str_agree():
    assert keyed_digest("x", key="k") == keyed_digest(b"x", key="k")


def test_empty_key_is_refused():
    """Falling back to an unkeyed hash would silently drop the guarantee."""
    with pytest.raises(HashKeyError):
        keyed_digest("prompt", key="")
