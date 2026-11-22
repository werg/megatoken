# Launch runtime-origin fixture repair

The full general CPU workflow at native `88764cc` installs a wheel. Its runtime-origin regression incorrectly assumed that `semantic_launch.__file__` lived under a source checkout and treated the wheel's Python library directory as the supplied consumer repository. The strict production runtime-origin check correctly refused this. Both Python CPU jobs failed the same test; the complete Python3.12 failure log is retained in `../v8-census-2026-11-22/`.

The regression now checks the actual import layout, explicitly simulates a wheel for the negative control, and explicitly simulates an imported source module for clean/dirty checkout controls. Production code and its strict runtime-origin validation are unchanged. A separate actual-data assertion compares the qualifier's V8 stream digest with an independently constructed ordinary production `MixtureWindowStream` using the same recipe, window, meter, rank, world size, and independent mode.

Local validation on the working increment based on `be9f446`:

- Full launch file: eleven controls pass; the actual-model checkpoint control fails because Torch is unavailable. The complete failure is retained and is not converted to a passing skip.
- Explicit non-ML selection: **11 passed, one deselected** (`-k 'not actual_model_checkpoint'`). The deselected actual-model check remains required in hosted CI.
- V8 export without legacy clients: **3 passed**.
- Ruff, Python compilation, and whitespace validation pass.

The small `uniseg==0.10.1` dependency was installed from the repository's exact hashed lock to execute the real prepared-corpus controls. Torch was not fabricated or substituted. These local selections overlap earlier tests and must not be added to the hosted baseline counts. Fresh general CPU and paired V8 results are still required for the published repair.

Decode each complete log with `base64 -d < FILE | gzip -d`; the manifest retains stored and decoded hashes. These checks perform no optimizer updates and do not establish learned task quality.
