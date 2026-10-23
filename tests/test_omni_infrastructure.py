from megatoken.controls import effective_event_payloads


def test_distinct_event_requests_are_causal_and_repeatable():
    prompts = effective_event_payloads(
        b"global", ("speech", "caption"), (("Speak English", 2), ("Describe visuals", 7))
    )
    assert prompts[0] != prompts[1]
    assert prompts == effective_event_payloads(
        b"global", ("speech", "caption"), (("Speak English", 2), ("Describe visuals", 7))
    )
    assert b"Describe visuals" not in prompts[0]
    peers = effective_event_payloads(b"", ("same", "same"), (("a", 0), ("b", 1)))
    assert peers[0] != peers[1]
