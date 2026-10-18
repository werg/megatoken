"""Lossless, bounded parsing evidence for semantic chunk planning.

Parsers describe original byte ranges. They never own tokenization or render a
replacement for the source; the planner is the only component that selects cuts.
"""
