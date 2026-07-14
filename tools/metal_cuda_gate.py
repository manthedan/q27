#!/usr/bin/env python3
"""Compare the canonical Metal generation with a running CUDA q27 server.

The CUDA server may bind loopback only; --cuda-ssh runs curl on that host.
This gate deliberately compares committed decoded text rather than floating-point
bit identity across architectures.
"""
import argparse, hashlib, json, subprocess, sys, urllib.request

CANONICAL_PROMPT = "The capital of France is"


def cuda_completion(args):
    body = json.dumps({"model": "qwen36-27b-mtp", "prompt": args.prompt,
                       "max_tokens": args.tokens, "temperature": 0}).encode()
    if args.cuda_ssh:
        command = ["ssh", args.cuda_ssh, "curl", "-fsS", args.cuda_url,
                   "-H", "Content-Type:application/json", "--data-binary", "@-"]
        result = subprocess.run(command, input=body, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=args.timeout, check=True)
        payload = result.stdout
    else:
        request = urllib.request.Request(args.cuda_url, data=body,
                                         headers={"Content-Type": "application/json"})
        payload = urllib.request.urlopen(request, timeout=args.timeout).read()
    return json.loads(payload)["choices"][0]["text"]


def metal_completion(args):
    command = [args.metal_binary, args.model, args.tokenizer, "--prompt", args.prompt,
               "-n", str(args.tokens), "--ctx", str(args.context)]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=args.timeout, check=True)
    prefix = b"generated:"
    if not result.stdout.startswith(prefix):
        raise RuntimeError("Metal CLI did not emit generated: output\n" +
                           result.stderr.decode(errors="replace"))
    generated = result.stdout[len(prefix):]
    if generated.endswith(b"\n"): generated = generated[:-1]  # printf framing only
    return generated.decode("utf-8", errors="replace"), result.stderr.decode(errors="replace")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("tokenizer")
    parser.add_argument("--metal-binary", default="./build/q27-metal")
    parser.add_argument("--cuda-url", default="http://127.0.0.1:18080/v1/completions")
    parser.add_argument("--cuda-ssh", help="SSH host when CUDA URL is remote loopback")
    parser.add_argument("--prompt", default=CANONICAL_PROMPT)
    parser.add_argument("-n", "--tokens", type=int, default=16)
    parser.add_argument("--ctx", dest="context", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error("token count is invalid")

    cuda = cuda_completion(args)
    metal, diagnostics = metal_completion(args)
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    print("CUDA :", repr(cuda))
    print("Metal:", repr(metal))
    print("CUDA sha256 :", digest(cuda))
    print("Metal sha256:", digest(metal))
    for line in diagnostics.splitlines():
        if "tokens in" in line or "model ready" in line:
            print(line)
    if cuda != metal:
        print("FAIL: committed decoded tokens differ", file=sys.stderr)
        return 1
    print("PASS: CUDA and Metal committed text match exactly")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (subprocess.SubprocessError, OSError, ValueError, KeyError) as error:
        print(f"gate error: {error}", file=sys.stderr)
        raise SystemExit(2)
