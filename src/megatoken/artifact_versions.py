"""Current artifact contracts, available to lightweight collection clients.

Writers, readers, and backup collection share these versions. This module must
remain standard-library only so collecting committed output never loads the
neural runtime. A version change rejects older artifacts; it does not migrate
or infer missing saved fields.
"""

PREPARED_SCHEMA_VERSION = 10
ARCHITECTURE_VERSION = 19
CHECKPOINT_FORMAT_VERSION = 22
