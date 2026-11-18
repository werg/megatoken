# Source binding capture fixture — production V6

`source-binding-capture.json` is a fresh execution of the pinned VS Code
`distinct` snippet with input `[1,2,1,0,2,5]`, collected by
`tracing-js-self/tools/generate-native-regression-fixtures.mjs`. It preserves
the closure's source binding names `seen` and `keyFn` in the production
`semantic-trace/6` language. The earlier V5 fixture remains historical evidence;
its trace and admission receipt were not translated into this fixture.

Source: `microsoft/vscode`, revision
`2bc7d5ab6919c62e60379e5e1f840bd80846dbdb`,
`src/vs/base/common/arrays.ts`. The full original file SHA-256 is
`e9a08190d445f28485a24c8b2701b5d91fda0702f3999f96d53f0cfebc032c0a`.
The source notice below also applies to the compact form in
`captured-parameter-name.json`.

Fresh collection passes owned replay and independent outcome, primitive,
branch and source-completion checks before exporting this public component
fixture. The producer preserves actual parameter source spans so implicit
argument initialization cannot invent a receipt for the first body declaration.

Private admission evidence is deliberately omitted. The exported sidecar
therefore declares controlled, locally verified, reference-unverified data.
Production-tokenizer tests check framing, supervision, visibility, source
binding identities and unchanged public bytes; full-source qualification and
learned execution require their separate receipts.

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
