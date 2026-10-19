"""Causal data contracts, streaming preparation, and exact stage accounting.

The data package has no PyTorch dependency. Codec implementations remain in
``megatoken.generation.codec``; this package uses only their lightweight protocol.
"""

from .accounting import Budget, TokenMeters
from .adapters import PROFILES, AdapterContext, SchemaError, adapt_row
from .chunking import (
    CHUNKING_VERSION,
    ChunkingConfig,
    ChunkingReport,
    ChunkingStats,
    SemanticChunk,
    segment_bytes,
    segment_with_report,
)
from .collection import (
    CollectionConfig,
    CollectionLedger,
    CollectionLimit,
    Collector,
    OpenAICompatibleTeacher,
    TaskSeed,
    Teacher,
    TeacherCall,
    TeacherRequest,
    TeacherResponse,
)
from .dedup import DedupConfig, DedupStore
from .demo import make_demo_episodes, write_demo_source
from .packing import TrainingWindow, WindowAudit, WindowConfig, iter_windows
from .parsing.types import ParseLimits
from .prepare import prepare_sources
from .recipes import StageRecipe, load_recipes, stage_recipe
from .review import ParserReviewPolicy, ParserReviewQueue
from .schema import (
    ByteSpan,
    Codec,
    Dependency,
    EncodedRecord,
    Episode,
    Lineage,
    LogicalAction,
    PhysicalRecord,
    SourceSpan,
    Supervision,
    apply_recovery_policy,
    encode_record,
    make_records,
)
from .sources import (
    SourceFile,
    SourceSpec,
    TermsReview,
    iter_source_rows,
    load_source_manifest,
    pin_hf_source,
    pin_local_source,
    write_source_manifest,
)
from .storage import PreparedDataset
from .student import ModelTeacher

__all__ = [
    "AdapterContext",
    "Budget",
    "ByteSpan",
    "Codec",
    "CHUNKING_VERSION",
    "ChunkingConfig",
    "ChunkingReport",
    "ChunkingStats",
    "SemanticChunk",
    "segment_with_report",
    "DedupConfig",
    "DedupStore",
    "Dependency",
    "EncodedRecord",
    "Episode",
    "Lineage",
    "LogicalAction",
    "ModelTeacher",
    "ParseLimits",
    "ParserReviewPolicy",
    "ParserReviewQueue",
    "PhysicalRecord",
    "PreparedDataset",
    "PROFILES",
    "SchemaError",
    "SourceFile",
    "SourceSpan",
    "SourceSpec",
    "StageRecipe",
    "Supervision",
    "TermsReview",
    "TokenMeters",
    "TrainingWindow",
    "WindowAudit",
    "WindowConfig",
    "adapt_row",
    "apply_recovery_policy",
    "encode_record",
    "iter_source_rows",
    "iter_windows",
    "load_recipes",
    "load_source_manifest",
    "make_demo_episodes",
    "make_records",
    "pin_hf_source",
    "pin_local_source",
    "prepare_sources",
    "segment_bytes",
    "stage_recipe",
    "write_demo_source",
    "write_source_manifest",
    "CollectionConfig",
    "CollectionLedger",
    "CollectionLimit",
    "Collector",
    "OpenAICompatibleTeacher",
    "TaskSeed",
    "Teacher",
    "TeacherCall",
    "TeacherRequest",
    "TeacherResponse",
]
