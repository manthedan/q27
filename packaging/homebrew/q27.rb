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
  url "https://github.com/manthedan/q27/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "42877ca9042daacfa5d580339da1444b61ee4923c41c17735a68b986ba5e1d07"
  license "MIT"

  depends_on :macos
  depends_on arch: :arm64

  def install
    shader = pkgshare/"q27_kernels.metal"
    ENV["CXXFLAGS"] =
      %Q{-O2 -std=c++17 -Wall -Wextra -DQ27_SHADER_PATH='"#{shader}"'}
    system "make", "build/q27-metal", "build/q27-metal-server",
           "build/tokenize_to_bin"
    bin.install "build/q27-metal"
    bin.install "build/q27-metal-server"
    bin.install "build/tokenize_to_bin" => "q27-tokenize"
    pkgshare.install "src/metal/q27_kernels.metal"
    doc.install "README.md", "docs/METAL_PROGRESS.md"
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
    EOS
  end

  test do
    # No model needed: the CLI prints usage and exits 1 without arguments,
    # which also proves the binary links and launches.
    assert_match "usage:", shell_output("#{bin}/q27-metal 2>&1", 1)
    assert_match "usage:", shell_output("#{bin}/q27-metal-server 2>&1", 1)
  end
end
