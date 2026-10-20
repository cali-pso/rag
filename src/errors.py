"""Errors raised on purpose, reported without a traceback."""


class RagError(Exception):
    """Base class of every error this project raises on purpose.

    The CLI catches it and prints its message instead of a traceback.
    """


class ErrorChunking(RagError):
    """A file could not be split into chunks."""


class UnknownQuestionID(RagError):
    """A ground-truth question has no match in the student results."""


class IndexNotFound(RagError):
    """The persisted index is missing: `index` was never run."""


class CorruptedIndex(RagError):
    """The persisted index exists but cannot be read back."""


class InvalidInputFile(RagError):
    """A user-supplied JSON file is malformed or has the wrong schema."""


class InvalidArgument(RagError):
    """A CLI argument has the wrong type or an out-of-range value."""
