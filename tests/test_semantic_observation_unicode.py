import json

from megatoken.controls import canonical_semantic_observation


def test_semantic_observation_preserves_unicode_and_escapes_surrogates():
    paired = chr(0xD834) + chr(0xDD1E)
    observation = {
        "astral": "𝄞",
        "cafe": "café",
        "lone": chr(0xD800),
        "pair": paired,
    }

    encoded = canonical_semantic_observation(observation)

    assert b'"cafe":"caf\xc3\xa9"' in encoded
    assert "𝄞".encode() in encoded
    assert b'"lone":"\\ud800"' in encoded
    assert b'"pair":"\\ud834\\udd1e"' in encoded
    decoded = json.loads(encoded.decode("utf-8"))
    assert decoded["lone"] == observation["lone"]
    assert decoded["pair"].encode("utf-16-le", "surrogatepass") == paired.encode(
        "utf-16-le", "surrogatepass"
    )
