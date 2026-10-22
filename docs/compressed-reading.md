# Compressed reading

Long text that the model only needs to *read* — a reference document, a large
tool result — would otherwise occupy many byte-level events of about 512
tokens each. Compressed reading feeds it instead through the same path as media:
the pinned EmbeddingGemma2 teacher's per-token hidden states (`[T, 512]`, up to
8192 teacher tokens per event), reduced by E's shared hierarchy into one
32-facet event. A 30 KB log becomes two events instead of fifteen.

- **What is compressed.** Observed fields (roles `tool` and `document` by
  default) of at least `min_bytes` that decode as UTF-8. Anything the model
  generates stays bytes; compressed text is input-only and a scored compressed
  record is rejected.
- **How.** `data/compression.py` (`DocumentCompressor`, `CompressionPolicy`)
  splits the field at line or word breaks into pieces of at most `piece_bytes`
  (halving any piece the teacher still finds over 8192 tokens), encodes each
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
