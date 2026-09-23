"""Small helpers for paths, directories, file I/O, and tokenization."""

import re
from pathlib import Path

import pandas as pd


def ensure_dir(path):
    """Create a directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def save_text_file(path, content, encoding="utf-8"):
    """Write text to a file, creating parent directories as needed."""
    ensure_dir(path.parent)
    path.write_text(content, encoding=encoding)


def load_csv(path, **kwargs):
    """Load a CSV file; forwards extra arguments to pandas.read_csv."""
    return pd.read_csv(path, **kwargs)


def file_nonempty(path):
    """True if path exists, is a file, and has size > 0."""
    return path.is_file() and path.stat().st_size > 0


# =============================
# Custom Tokenizer
# =============================
# Defined here in utils.py so it can be imported by both train.py and
# evaluate.py. This is required because joblib pickles the tokenizer
# function reference - if it lives in train.py's __main__, evaluate.py
# cannot load the saved pipeline.
def sqli_tokenizer(text):
    """
    Splits text into alphanumeric tokens AND treats SQL-specific special
    characters as distinct tokens. This is the "Specialized Feature
    Engineering" claim from Section 2.4 of the research paper.
    """
    pattern = r"[a-zA-Z0-9_]+|['\"\-;/*#=()%<>]"
    return re.findall(pattern, str(text).lower())