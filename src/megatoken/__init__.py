"""Megatoken's lightweight package entry point.

Data preparation deliberately does not import PyTorch. Neural modules are loaded
by training and generation commands, after configuration has been validated.
"""

__version__ = "0.1.0"
