from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SearchRequest(_message.Message):
    __slots__ = ("query", "top_k")
    QUERY_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    query: str
    top_k: int
    def __init__(self, query: _Optional[str] = ..., top_k: _Optional[int] = ...) -> None: ...

class Passage(_message.Message):
    __slots__ = ("chunk_id", "doc_id", "title", "text", "score")
    CHUNK_ID_FIELD_NUMBER: _ClassVar[int]
    DOC_ID_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float
    def __init__(self, chunk_id: _Optional[str] = ..., doc_id: _Optional[str] = ..., title: _Optional[str] = ..., text: _Optional[str] = ..., score: _Optional[float] = ...) -> None: ...

class SearchResponse(_message.Message):
    __slots__ = ("passages",)
    PASSAGES_FIELD_NUMBER: _ClassVar[int]
    passages: _containers.RepeatedCompositeFieldContainer[Passage]
    def __init__(self, passages: _Optional[_Iterable[_Union[Passage, _Mapping]]] = ...) -> None: ...

class UpsertDocumentRequest(_message.Message):
    __slots__ = ("title", "text")
    TITLE_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    title: str
    text: str
    def __init__(self, title: _Optional[str] = ..., text: _Optional[str] = ...) -> None: ...

class UpsertDocumentResponse(_message.Message):
    __slots__ = ("doc_id", "title", "chunks", "created")
    DOC_ID_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    CHUNKS_FIELD_NUMBER: _ClassVar[int]
    CREATED_FIELD_NUMBER: _ClassVar[int]
    doc_id: str
    title: str
    chunks: int
    created: bool
    def __init__(self, doc_id: _Optional[str] = ..., title: _Optional[str] = ..., chunks: _Optional[int] = ..., created: _Optional[bool] = ...) -> None: ...

class ListDocumentsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class DocumentInfo(_message.Message):
    __slots__ = ("doc_id", "title", "chunks")
    DOC_ID_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    CHUNKS_FIELD_NUMBER: _ClassVar[int]
    doc_id: str
    title: str
    chunks: int
    def __init__(self, doc_id: _Optional[str] = ..., title: _Optional[str] = ..., chunks: _Optional[int] = ...) -> None: ...

class ListDocumentsResponse(_message.Message):
    __slots__ = ("documents",)
    DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    documents: _containers.RepeatedCompositeFieldContainer[DocumentInfo]
    def __init__(self, documents: _Optional[_Iterable[_Union[DocumentInfo, _Mapping]]] = ...) -> None: ...

class DeleteDocumentRequest(_message.Message):
    __slots__ = ("doc_id",)
    DOC_ID_FIELD_NUMBER: _ClassVar[int]
    doc_id: str
    def __init__(self, doc_id: _Optional[str] = ...) -> None: ...

class DeleteDocumentResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
