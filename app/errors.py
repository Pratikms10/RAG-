"""Typed errors converted to appropriate HTTP responses by the API layer."""


class RAGError(Exception):
    """Base class for expected application errors."""


class UnsupportedDocumentError(RAGError):
    pass


class DocumentParseError(RAGError):
    pass


class EmptyDocumentError(RAGError):
    pass


class DuplicateDocumentError(RAGError):
    pass


class EmbeddingProviderError(RAGError):
    pass


class IndexCompatibilityError(RAGError):
    pass
