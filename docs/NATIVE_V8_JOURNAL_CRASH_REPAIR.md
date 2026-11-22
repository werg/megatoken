# Recovering immutable request publication after process exit

An immutable publication writes and fsyncs a private temporary file, links its final record name, removes the temporary name, and fsyncs the containing directory. A process exit between the link and unlink leaves two names for one valid inode. Rejecting every two-link temporary made that legitimate crash boundary unreadable.

Recovery holds the existing native scope lock and repairs only an owner-private regular temporary inode with one link, or exactly two links and one matching official intent, outcome or superseded record in that scope. An extra alias or missing official destination is refused without deleting either name. Cleanup is directory-synced; the ordinary record reader independently fsyncs and validates the original bytes. The producer counterpart applies the same narrow repair, including direct ordinal inspection, and bounds total retained intent bytes separately from per-record size.

Real subprocess exits exercise native intent, outcome and superseded-marker publication and producer intent publication. An intent without an outcome remains unknown and is never redispatched; an exact completed outcome remains readable. A durable superseded marker still forbids restoring the older state even when its original publication acknowledgement was lost. These are storage/ownership guarantees, not permission for automatic host replay.

Qualification is isolated from the main PR branches until the paired hosted runtime, lint, formatting and zero-skip gates pass. The base implementations are producer e7f8fd60b6f286a667ddf708e17a0963aec78211 and native abf4dc6424e3aeaf10f1d2609a2197897c1d42ba.
