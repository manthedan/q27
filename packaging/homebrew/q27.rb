# Homebrew formula for the q27 Metal engine (packaging/homebrew/q27.rb).
# Install directly:  brew install --formula ./packaging/homebrew/q27.rb
# or from a tap that carries this file as Formula/q27.rb.
#
# Ships the Metal CLI, the OpenAI/Anthropic-compatible server, and the
# corpus tokenizer. Model artifacts and tokenizers are NOT included (see
# caveats). The Metal shader compiles from source at runtime; the formula
# installs it to pkgshare and bakes that path into the binaries
# (Q27_SHADER_PATH), so they run from any directory.
class Q27 < Formula
  desc "Ternary-quantized 27B LLM inference engine for Apple silicon (Metal)"
  homepage "https://github.com/manthedan/q27"
  url "https://github.com/manthedan/q27/archive/refs/tags/metal-v0.5.0.tar.gz"
  sha256 "4751967c628b3279f1bde885eec1b5ce95ae45b0b0e5f5042d257d3e24b7425e"
  # The metal-v* tag line is independent of upstream's v0.3.x CUDA tags; set
  # the version explicitly since the "metal-v" prefix doesn't parse to one.
  version "0.5.0"
  license "MIT"

  depends_on :macos
  depends_on arch: :arm64

  def install
    shader = pkgshare/"q27_kernels.metal"
    ENV["CXXFLAGS"] =
      %Q{-O2 -std=c++17 -Wall -Wextra -DQ27_SHADER_PATH='"#{shader}"'}
    system "make", "build/q27-metal", "build/q27-metal-server",
           "build/tokenize_to_bin",
           "build/metal_decode_bench", "build/metal_prefill_bench",
           "build/q27-agent"
    bin.install "build/q27-metal"
    bin.install "build/q27-metal-server"
    bin.install "build/tokenize_to_bin" => "************"
    bin.install "build/metal_decode_bench"
    bin.install "build/metal_prefill_bench"
    # The experimental native agent (durable sessions + compaction). It links
    # the engine directly and inherits the baked Q27_SHADER_PATH, so it runs
    # from any directory like the other binaries. See caveats.
    bin.install "build/q27-agent"
    pkgshare.install "src/metal/q27_kernels.metal"

    # The supervisor wrapper, bench/report tools, model registry, and the
    # python repack/tokenizer tools. These scripts resolve their dependencies
    # RELATIVE to their own location (bin/../lib, bin/../models.tsv,
    # ../share/q27-tools, and a dev build/ dir). To keep that resolution
    # intact under Homebrew, install the whole tree under libexec/q27 in the
    # same shape as packaging/, then symlink the entry points into bin/.
    (libexec/"q27/bin").install Dir["packaging/bin/*"]
    (libexec/"q27/lib").install "packaging/lib/q27_bench_lib.sh"
    (libexec/"q27").install "packaging/models.tsv"
    (libexec/"q27/share/q27-tools").install "tools/repack.py",
                                           "tools/export_tokenizer.py"
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
    doc.install "README.md", "docs/METAL_PROGRESS.md", "docs/MODELS.md"
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

      EXPERIMENTAL: q27-agent is the native agent (durable sessions +
      compaction, no server). It is a Phase-0 experiment, not a finished
      product:

        q27-agent MODEL.q27 MODEL.tok --session work.q27agent
    EOS
  end

  test do
    # No model needed: the CLI prints usage and exits 1 without arguments,
    # which also proves the binary links and launches.
    assert_match "usage:", shell_output("#{bin}/q27-metal 2>&1", 1)
    assert_match "usage:", shell_output("#{bin}/q27-metal-server 2>&1", 1)
    # q27-agent prints usage to stderr and exits 2 with no model/tokenizer.
    assert_match "usage:", shell_output("#{bin}/q27-agent 2>&1", 2)
  end
end
