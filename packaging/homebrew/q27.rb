# Homebrew formula for the q27 Metal engine (packaging/homebrew/q27.rb).
# Install directly:  brew install --formula ./packaging/homebrew/q27.rb
# or from a tap that carries this file as Formula/q27.rb.
#
# Ships PREBUILT macOS arm64 binaries (the GoReleaser-pattern: binaries are
# built and gated on the release machine, attached to the GitHub release,
# and this formula downloads the tarball — no toolchain needed at install
# time, not even CLT). Ships the Metal CLI, the OpenAI/Anthropic-compatible
# server, the native agent, the Ratatui TUI, and the corpus tokenizer.
# Model artifacts and tokenizers are NOT included (see caveats). The Metal
# shader compiles from source at runtime; the formula installs it to
# pkgshare and the binaries carry that baked path (Q27_SHADER_PATH), so
# they run from any directory.
class Q27 < Formula
  desc "Ternary-quantized 27B LLM inference engine for Apple silicon (Metal)"
  homepage "https://github.com/manthedan/q27"
  url "https://github.com/manthedan/q27/releases/download/metal-v0.6.0/q27-metal-v0.6.0-macos-arm64.tar.gz"
  sha256 "633d64e13706ff4017c1c9da39cbc63181d9aac6cb7017d28f8f9a000ed6ee47"
  license "MIT"

  depends_on arch: :arm64
  depends_on macos: :ventura

  def install
    # Prebuilt binaries (built with MACOSX_DEPLOYMENT_TARGET=13.0).
    %w[q27-metal q27-metal-server q27-agent q27-tui tokenize_to_bin
       metal_decode_bench metal_prefill_bench].each do |b|
      bin.install "bin/#{b}"
    end
    bin.install "bin/tokenize_to_bin" => "q27-tokenize"
    pkgshare.install "share/q27_kernels.metal"

    # The supervisor wrapper, bench/report tools, model registry, and the
    # python repack/tokenizer tools. These scripts resolve their dependencies
    # RELATIVE to their own location (bin/../lib, bin/../models.tsv,
    # ../share/q27-tools, and a dev build/ dir). To keep that resolution
    # intact under Homebrew, install the whole tree under libexec/q27 in the
    # same shape as packaging/, then symlink the entry points into bin/.
    (libexec/"q27/bin").install Dir["packaging/bin/*"]
    (libexec/"q27/lib").install "packaging/lib/q27_bench_lib.sh"
    (libexec/"q27").install "packaging/models.tsv"
    (libexec/"q27/share/q27-tools").install "share/q27-tools/repack.py",
                                           "share/q27-tools/export_tokenizer.py"
    # The wrapper resolves its lib/models/tools relative to its own real
    # path, but its symlink-chase uses `dirname "$0"` rather than `dirname
    # "$SRC"`, so it breaks on Homebrew's nested symlink chain
    # (/opt/homebrew/bin/q27 -> Cellar/bin/q27 -> ../libexec/q27/bin/q27).
    # Install bin/q27 as a tiny shim that execs the libexec wrapper directly,
    # sidestepping the resolution. The leaf tools are simple enough to keep as
    # symlinks.
    (bin/"q27").write <<~SH
      #!/bin/sh
      exec "#{libexec}/q27/bin/q27" "$@"
    SH
    (bin/"q27").chmod 0555
    %w[q27-bench q27-report q27-fetch].each do |t|
      bin.install_symlink libexec/"q27/bin"/t
    end
    doc.install Dir["doc/*"]
  end

  def caveats
    <<~EOS
      q27 needs a model artifact (.q27) and tokenizer (.tok), which are not
      distributed with this formula. Repack instructions and supported
      checkpoints: https://github.com/manthedan/q27#weights

        q27-metal MODEL.q27 TOKENIZER.tok --prompt "..." -n 64
        q27-metal-server MODEL.q27 TOKENIZER.tok --port 8080

      Chunked prefill and speculative decoding need an Apple7+ GPU family
      device (M1 or newer). 16 GB unified memory is a practical minimum for
      the 27B ternary artifact.

      The agent defaults to the Ratatui TUI on a terminal (Q27_AGENT_UI=auto);
      Q27_AGENT_UI=classic keeps the linenoise UI:

        q27 agent MODEL.q27 MODEL.tok --session work.q27agent

      Binaries find the Metal shader via the brewed share path; a custom
      source tree can be forced with Q27_METAL_SOURCE=/path/to/q27_kernels.metal.
    EOS
  end

  test do
    # No model needed: the CLI prints usage and exits 1 without arguments,
    # which also proves the binary links and launches.
    assert_match "usage:", shell_output("#{bin}/q27-metal 2>&1", 1)
    assert_match "usage:", shell_output("#{bin}/q27-metal-server 2>&1", 1)
    # q27-agent prints usage to stderr and exits 2 with no model/tokenizer.
    assert_match "usage:", shell_output("#{bin}/q27-agent 2>&1", 2)
    # The Ratatui TUI prints usage and exits 0 with no model/tokenizer.
    assert_match "usage:", shell_output("#{bin}/q27-tui 2>&1")
  end
end
