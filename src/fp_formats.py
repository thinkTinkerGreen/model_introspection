import math
import struct
import torch

def round_to_format(x: float, exp_bits: int, man_bits: int, bias: int) -> int:
    """
    Encode float x into unsigned integer bit representation of (1, exp_bits, man_bits)
    with given bias, using round-to-nearest-even.
    Returns integer bit pattern.
    """
    if x == 0.0:
        return 0
    sign = 0
    if x < 0:
        sign = 1
        x = -x

    m, e_raw = math.frexp(x)
    m = m * 2.0
    e = e_raw - 1  # unbiased exponent

    biased_e = e + bias

    if biased_e >= 1:
        # Normal representation: significand is 1.frac
        frac = m - 1.0
        scaled = frac * (2 ** man_bits)
        base = int(scaled)
        rem = scaled - base

        # Round to nearest even
        if rem > 0.5:
            rounded = base + 1
        elif rem < 0.5:
            rounded = base
        else:
            if base % 2 == 1:
                rounded = base + 1
            else:
                rounded = base

        if rounded == (1 << man_bits):
            rounded = 0
            biased_e += 1

        return (sign << (exp_bits + man_bits)) | (biased_e << man_bits) | rounded
    else:
        # Subnormal representation: significand is 0.frac
        shift = 1 - biased_e
        scaled = (m / (2 ** shift)) * (2 ** man_bits)
        base = int(scaled)
        rem = scaled - base

        if rem > 0.5:
            rounded = base + 1
        elif rem < 0.5:
            rounded = base
        else:
            if base % 2 == 1:
                rounded = base + 1
            else:
                rounded = base

        if rounded == (1 << man_bits):
            rounded = 0
            biased_e = 1

        return (sign << (exp_bits + man_bits)) | (biased_e << man_bits) | rounded

def decode_format(bits: int, exp_bits: int, man_bits: int, bias: int) -> float:
    """
    Decode an unsigned integer bit representation of (1, exp_bits, man_bits)
    with given bias back to float.
    """
    sign_mask = 1 << (exp_bits + man_bits)
    sign = -1.0 if (bits & sign_mask) else 1.0

    exp_mask = ((1 << exp_bits) - 1) << man_bits
    biased_e = (bits & exp_mask) >> man_bits

    man_mask = (1 << man_bits) - 1
    man = bits & man_mask

    if biased_e == 0:
        # Subnormal
        if man == 0:
            return 0.0 * sign
        frac = man / (2 ** man_bits)
        return sign * frac * (2 ** (1 - bias))
    else:
        frac = 1.0 + man / (2 ** man_bits)
        return sign * frac * (2 ** (biased_e - bias))

def get_format_table_0_1():
    """
    Generates dictionary of exact representations of 0.1 across formats.
    """
    records = {}

    # fp32
    f32_bytes = struct.pack(">f", 0.1)
    f32_int = struct.unpack(">I", f32_bytes)[0]
    f32_val = struct.unpack(">f", f32_bytes)[0]
    f32_s = (f32_int >> 31) & 1
    f32_e = (f32_int >> 23) & 0xFF
    f32_m = f32_int & 0x7FFFFF
    records["fp32"] = {
        "bits": f"{f32_s} {f32_e:08b} {f32_m:023b}",
        "hex": f"0x{f32_int:08X}",
        "stored_value": f32_val,
        "abs_error": abs(f32_val - 0.1),
        "rel_error": abs(f32_val - 0.1) / 0.1,
        "spacing": 2 ** (-27),
    }

    # bf16
    bf16_val = torch.tensor(0.1, dtype=torch.float32).to(torch.bfloat16)
    bf16_int = bf16_val.view(torch.int16).item() & 0xFFFF
    bf16_float = bf16_val.to(torch.float32).item()
    bf16_s = (bf16_int >> 15) & 1
    bf16_e = (bf16_int >> 7) & 0xFF
    bf16_m = bf16_int & 0x7F
    records["bf16"] = {
        "bits": f"{bf16_s} {bf16_e:08b} {bf16_m:07b}",
        "hex": f"0x{bf16_int:04X}",
        "stored_value": bf16_float,
        "abs_error": abs(bf16_float - 0.1),
        "rel_error": abs(bf16_float - 0.1) / 0.1,
        "spacing": 2 ** (-11),
        "truncation_hex": "0x3DCC",
        "truncation_value": 0.099609375,
    }

    # fp8 E4M3 (E4M3FN)
    fp8_val = torch.tensor(0.1, dtype=torch.float32).to(torch.float8_e4m3fn)
    fp8_int = fp8_val.view(torch.uint8).item()
    fp8_float = decode_format(fp8_int, exp_bits=4, man_bits=3, bias=7)
    fp8_s = (fp8_int >> 7) & 1
    fp8_e = (fp8_int >> 3) & 0x0F
    fp8_m = fp8_int & 0x07
    records["fp8_e4m3"] = {
        "bits": f"{fp8_s} {fp8_e:04b} {fp8_m:03b}",
        "hex": f"0x{fp8_int:02X}",
        "stored_value": fp8_float,
        "abs_error": abs(fp8_float - 0.1),
        "rel_error": abs(fp8_float - 0.1) / 0.1,
        "spacing": 2 ** (-7),
    }

    # fp8 E5M2 (bonus)
    e5m2_int = round_to_format(0.1, exp_bits=5, man_bits=2, bias=15)
    e5m2_float = decode_format(e5m2_int, exp_bits=5, man_bits=2, bias=15)
    e5m2_s = (e5m2_int >> 7) & 1
    e5m2_e = (e5m2_int >> 2) & 0x1F
    e5m2_m = e5m2_int & 0x03
    records["fp8_e5m2"] = {
        "bits": f"{e5m2_s} {e5m2_e:05b} {e5m2_m:02b}",
        "hex": f"0x{e5m2_int:02X}",
        "stored_value": e5m2_float,
        "abs_error": abs(e5m2_float - 0.1),
        "rel_error": abs(e5m2_float - 0.1) / 0.1,
        "spacing": 2 ** (-6),
    }

    return records

def run_accumulation_demo():
    """
    Demonstrates compounding rounding error when accumulating in bf16 vs fp32.
    """
    # fp32 accumulation
    s_fp32 = 0.0
    for _ in range(10):
        s_fp32 += 0.1

    # bf16 inputs with bf16 accumulation
    s_bf16_acc = torch.tensor(0.0, dtype=torch.bfloat16)
    x_bf16 = torch.tensor(0.1, dtype=torch.bfloat16)
    for _ in range(10):
        s_bf16_acc = s_bf16_acc + x_bf16

    # bf16 inputs with fp32 accumulation
    s_fp32_acc = torch.tensor(0.0, dtype=torch.float32)
    for _ in range(10):
        s_fp32_acc = s_fp32_acc + x_bf16.to(torch.float32)

    return {
        "exact": 1.0,
        "fp32_sum": s_fp32,
        "fp32_error": abs(s_fp32 - 1.0),
        "bf16_accum_sum": s_bf16_acc.to(torch.float32).item(),
        "bf16_accum_error": abs(s_bf16_acc.to(torch.float32).item() - 1.0),
        "bf16_in_fp32_accum_sum": s_fp32_acc.item(),
        "bf16_in_fp32_accum_error": abs(s_fp32_acc.item() - 1.0),
    }
