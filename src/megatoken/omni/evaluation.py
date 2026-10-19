"""Self-contained measurable baseline metrics, never fabricated model scores."""
from __future__ import annotations

def edit_distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        curr=[i]+[0]*len(b)
        for j,y in enumerate(b,1):
            curr[j]=min(curr[j-1]+1,prev[j]+1,prev[j-1]+(x!=y))
        prev=curr
    return prev[-1]

def word_error_rate(reference, hypothesis):
    gold=reference.split()
    return edit_distance(gold,hypothesis.split())/max(1,len(gold))

def character_error_rate(reference,hypothesis):
    return edit_distance(reference,hypothesis)/max(1,len(reference))

def exact_entity_recall(expected, realized):
    expected=tuple(expected)
    if not expected:
        return 1.
    return sum(str(entity) in realized for entity in expected)/len(expected)

def normalized_counter(numerator, denominator):
    if denominator <= 0:
        raise ValueError("Metric needs at least one valid site")
    return numerator/denominator
