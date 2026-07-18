# Third-party notice

The native-agent architecture and porting target in this experiment are based
on the DS4 project at:

- https://github.com/antirez/ds4
- pinned commit: `80ebbc396aee40eedc1d829222f3362d10fa4c6c`
- upstream file: `ds4_agent.c`

DS4 is distributed under the MIT License:

> MIT License
>
> Copyright (c) 2026 The ds4.c authors
> Copyright (c) 2023-2026 The ggml authors
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

The Phase-0 implementation does not vendor the complete upstream file. If
substantial DS4 source is copied during later phases, this notice must remain
with the derived source and release artifacts.
