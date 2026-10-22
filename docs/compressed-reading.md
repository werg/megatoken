# Compressed reading

Long text that the model only needs to *read* — a reference document, a large
tool result — would otherwise occupy many byte-level events of about 512
tokens each. Compressed reading feeds it instead through the same path as media:
the pinned EmbeddingGemma2 teacher's per-token hidden states (`[T, 512]`, up to
8192 teacher tokens per event), reduced by E's shared hierarchy into one
32-facet event. Measured with the real teacher: an 88 KB build-log-like
document becomes 5 events of ~8,180 teacher tokens each, where byte-level
reading would take about 44.

- **What is compressed.** Observed fields (roles `tool` and `document` by
  default) of at least `min_bytes` that decode as UTF-8. Anything the model
  generates stays bytes; compressed text is input-only and a scored compressed
  record is rejected.
- **How.** `data/compression.py` (`DocumentCompressor`, `CompressionPolicy`)
  splits the field by the teacher's own token count into pieces that each fill
  one teacher input (8192 tokens, prompt included; cut at a line break near the
  end when there is one; `EmbeddingGemma2.text_pieces`), encodes each
  piece with the teacher's fixed document prompt (`title: none | text: `), and
  stores the features as immutable assets. The episode carries one canonical
  descriptor record per piece (a `MediaRecord` of modality `text`, which also
  records the prompt) whose source span locates the original bytes. Its event
  modality is `compressed_text`, with its own modality embedding in C.
- **Where.** `megatoken prepare ... --compress-min-bytes 8192 --teacher-packs
  packs/lock.json` turns it on for every imported source; features are written
  under the prepared dataset root, which training reads as its media root.
  Imported episodes are grouped one record per meridian, one track per role.
- **Prompt.** Fixed for now, so a document always maps to the same features and
  is cached once. Question-aware encoding (the current question in the
  teacher prompt, which bgkit found carries most of the benefit) is the
  planned, measured upgrade; it needs the teacher live during training and
  serving.

## The long-document reading stream

`scripts/prepare_long_document_reading.py` builds the `long_document_reading`
stream that the stage plan mixes into base pretraining and later stages: the
user's question, the reference documents joined into one compressed document,
and the scored answer, which must come from that document. Tasks come from a
JSONL file (`{"id", "question", "documents", "answer"}`) and/or straight from
HotpotQA and MuSiQue, where every paragraph of a question (supporting and
distractor) is part of its document.
