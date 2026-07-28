#!/usr/bin/env python3
import pathlib
import struct
import subprocess
import sys
import tempfile

MAGIC = 0x46373251
VERSION = 1
ALIGN = 256

TENSORS = [
    ("token_embd.weight", 2, [1, 128], 128, 2),
    ("blk.0.ffn_gate.weight", 5, [1, 128], 26, 2),
    ("blk.3.attn_q.weight", 6, [1, 128], 16, 2),
    ("blk.64.nextn.eh_proj.weight", 4, [1, 128], 32, 2),
    ("output_norm.weight", 0, [4], 16, 0),
    ("test.f16", 1, [2], 4, 0),
    ("test.q4", 3, [1, 64], 32, 2),
]


def align(value):
    return (value + ALIGN - 1) // ALIGN * ALIGN


def make_fixture(path, corrupt_size=False, misaligned=False, bad_t2=False,
                 bad_t3_range=False, bad_t3_padding=False, overflow=False,
                 bad_offset=False, bad_embedding=False):
    meta = b'{}'
    entries = []
    blobs = bytearray()

    for index, (name, dtype, shape, data_size, scale_size) in enumerate(TENSORS):
        if bad_embedding and name == "token_embd.weight":
            dtype, data_size = 4, 32
        if misaligned and index == 0:
            shape = [1, 129]
        if overflow and dtype == 5:
            shape = [(1 << 63) + 1, 128]
        data_off = align(len(blobs))
        stored_data_off = (1 << 64) - 128 if bad_offset and dtype == 4 else data_off
        blobs.extend(b'\0' * (data_off - len(blobs)))
        stored_data_size = data_size + (1 if corrupt_size and index == 1 else 0)
        data = bytearray(bytes([index + 1]) * stored_data_size)
        if dtype == 5:
            data[25] = 108
            if bad_t3_range:
                data[0] = 243
            if bad_t3_padding:
                data[25] = 0
        if dtype == 4 and bad_t2:
            data[0] = 3
        blobs.extend(data)

        scale_off = 0
        if scale_size:
            scale_off = align(len(blobs))
            blobs.extend(b'\0' * (scale_off - len(blobs)))
            blobs.extend(b'\0<' * (scale_size // 2))

        name_bytes = name.encode()
        entry = bytearray(struct.pack('<H', len(name_bytes)))
        entry.extend(name_bytes)
        entry.extend(struct.pack('<BB', dtype, len(shape)))
        entry.extend(struct.pack('<' + 'Q' * len(shape), *shape))
        entry.extend(struct.pack('<QQQQ', stored_data_off, stored_data_size, scale_off, scale_size))
        entries.append(entry)

    header = struct.pack('<IIII', MAGIC, VERSION, len(entries), len(meta)) + meta
    table = b''.join(entries)
    data_base = align(len(header) + len(table))
    path.write_bytes(header + table + b'\0' * (data_base - len(header) - len(table)) + blobs)


def run(inspect, fixture):
    return subprocess.run([inspect, str(fixture)], text=True, capture_output=True)


def require_failure(inspect, fixture, message, failure):
    result = run(inspect, fixture)
    if result.returncode == 0 or message not in result.stdout + result.stderr:
        sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit(failure)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f'usage: {sys.argv[0]} build/inspect')
    inspect = sys.argv[1]
    with tempfile.TemporaryDirectory(prefix='q27-inspect-') as tmp:
        root = pathlib.Path(tmp)
        fixtures = {
            'valid': root / 'valid.q27',
            'corrupt': root / 'corrupt.q27',
            'misaligned': root / 'misaligned.q27',
            'bad_t2': root / 'bad-t2.q27',
            'bad_t3_range': root / 'bad-t3-range.q27',
            'bad_t3_padding': root / 'bad-t3-padding.q27',
            'overflow': root / 'overflow.q27',
            'bad_offset': root / 'bad-offset.q27',
            'bad_embedding': root / 'bad-embedding.q27',
        }
        make_fixture(fixtures['valid'])
        make_fixture(fixtures['corrupt'], corrupt_size=True)
        make_fixture(fixtures['misaligned'], misaligned=True)
        make_fixture(fixtures['bad_t2'], bad_t2=True)
        make_fixture(fixtures['bad_t3_range'], bad_t3_range=True)
        make_fixture(fixtures['bad_t3_padding'], bad_t3_padding=True)
        make_fixture(fixtures['overflow'], overflow=True)
        make_fixture(fixtures['bad_offset'], bad_offset=True)
        make_fixture(fixtures['bad_embedding'], bad_embedding=True)

        good = run(inspect, fixtures['valid'])
        if good.returncode != 0 or '\nOK\n' not in good.stdout:
            sys.stderr.write(good.stdout + good.stderr)
            raise SystemExit('valid packed-dtype fixture failed inspection')

        require_failure(inspect, fixtures['bad_embedding'],
                        'token_embd.weight: CUDA row lookup requires Q8_G128',
                        'non-Q8 CUDA embedding was not rejected')
        require_failure(inspect, fixtures['corrupt'],
                        'tensor byte-size mismatch: blk.0.ffn_gate.weight',
                        'one-byte packed-dtype corruption was not detected')
        require_failure(inspect, fixtures['misaligned'],
                        'Q8 columns not divisible by 128',
                        'misaligned packed-dtype shape was not detected')
        require_failure(inspect, fixtures['bad_t2'],
                        'T2 payload contains reserved code 3',
                        'reserved T2 code was not detected')
        require_failure(inspect, fixtures['bad_t3_range'],
                        'T3 payload byte exceeds 242',
                        'out-of-range T3 byte was not detected')
        require_failure(inspect, fixtures['bad_t3_padding'],
                        'T3 final-byte padding is noncanonical',
                        'noncanonical T3 padding was not detected')
        require_failure(inspect, fixtures['overflow'],
                        'size overflow',
                        'packed payload size overflow was not detected')
        require_failure(inspect, fixtures['bad_offset'],
                        'unaligned tensor blob',
                        'overflowing tensor offset was not detected')

    print('inspect packed dtype fixtures: PASS')


if __name__ == '__main__':
    main()
