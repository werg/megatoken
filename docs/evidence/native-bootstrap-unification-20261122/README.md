# Native bootstrap unification evidence

The two `prior56-*` logs are preserved from external candidate commit
`56cce3b80423b0cb70c3a096d867561066e3a211`. They record failures on that earlier
published native candidate, before the current tree adds the canonical child
bootstrap wire API and the 7f create-predecessor proof. They are retained as
historical reproduction evidence; they are not runs against this tree.

`prior56-pre-fix-response-reproduction.log` records a rehashed saved response
whose child parent became `forged-parent` while the postcheckpoint stayed the
same. `prior56-pre-fix-alias-reproduction.log` records the two mixed underscore
and hyphen spellings that passed the guarded raw dispatch before normalization.
The corrected union must independently pass the retained response and all-four
alias controls before it is qualified.
