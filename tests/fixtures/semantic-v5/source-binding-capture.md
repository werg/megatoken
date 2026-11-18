# Source binding capture fixture

`source-binding-capture.json` retains the exact public timeline reconstructed
from failed tracing-js-self candidate `real-utilities-5a1fde95b4daab8314e03d36`
with immutable producer `83e0d5742e63dc4439b68d6296f323cb7a3aafdc`. The pinned
VS Code `distinct` implementation closes over `seen` and `keyFn`; the latter
is a source binding name, not a heap reference identifier.

Source: `microsoft/vscode`, revision
`2bc7d5ab6919c62e60379e5e1f840bd80846dbdb`,
`src/vs/base/common/arrays.ts`. The full original file SHA-256 is
`e9a08190d445f28485a24c8b2701b5d91fda0702f3999f96d53f0cfebc032c0a`.

The original/owned independent source checks passed in the reconstruction.
Private evidence is omitted from this component-test fixture, so its sidecar
explicitly declares controlled, locally verified, reference-unverified data.
Production-tokenizer testing establishes native framing, masks, visibility,
and unchanged public bytes, not full source qualification or model learning.
The complete reconstruction and before/after native receipts are retained
separately in the tracing-js-self release evidence.

This fixes validation of an existing v5 source-binding relation; neither the
wire format nor contract version changes. Captures use the same bounded
source-name predicate as existing `name`, `bind`, and `from` fields. These
tests establish camel-case, underscore, lexical suffix, and dollar-sign
behavior, without asserting complete Unicode identifier parity. Heap
references retain their separate, stricter grammar.

## Original source notice

MIT License

Copyright (c) 2015 - present Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
