import torch
import triton
import triton.language as tl


@triton.jit
def _repetition_penalty_kernel(
    logits_ptr,
    prompt_mask_ptr,
    output_mask_ptr,
    penalties_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid is (num_vocab_blocks, num_seqs): program_id(0) is the fastest-varying
    # axis, so programs scheduled consecutively on the same CPU thread (OpenMP
    # static scheduling) access contiguous memory instead of striding across
    # rows.
    vocab_block = tl.program_id(0)
    seq_idx = tl.program_id(1)

    vocab_idx = vocab_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    idx = seq_idx * vocab_size + vocab_idx

    valid_vocab = vocab_idx < vocab_size
    prompt_mask = tl.load(prompt_mask_ptr + idx, mask=valid_vocab, other=0)
    output_mask = tl.load(output_mask_ptr + idx, mask=valid_vocab, other=0)
    logits = tl.load(logits_ptr + idx, mask=valid_vocab, other=0.0)

    is_repeated = (prompt_mask | output_mask) != 0

    penalty = tl.load(penalties_ptr + seq_idx)

    out = tl.where(is_repeated & (logits > 0), logits / penalty, logits)
    out = tl.where(is_repeated & (logits <= 0), logits * penalty, out)

    tl.store(logits_ptr + idx, out, mask=valid_vocab)


def _select_block_size(vocab_size):
    # Prefer an EVEN (unmasked) kernel: pick the largest power-of-two block
    # (up to 1024) that divides vocab_size.
    for block in (1024, 512, 256, 128):
        if vocab_size % block == 0:
            return block
    return 1024


def apply_repetition_penalties(logits, prompt_mask, output_mask, repetition_penalties):
    assert logits.is_contiguous(), "logits must be contiguous"
    assert (
        prompt_mask.is_contiguous() and prompt_mask.dtype == torch.bool
    ), "prompt_mask must be contiguous bool tensor"
    assert (
        output_mask.is_contiguous() and output_mask.dtype == torch.bool
    ), "output_mask must be contiguous bool tensor"
    assert (
        repetition_penalties.is_contiguous()
    ), "repetition_penalties must be contiguous"
    assert logits.dim() == 2, f"logits must be 2D, got {logits.dim()}D"
    assert (
        logits.shape == prompt_mask.shape == output_mask.shape
    ), "shape mismatch between logits and masks"
    assert (
        repetition_penalties.dim() == 1
        and repetition_penalties.numel() == logits.shape[0]
    ), "repetition_penalties must be 1D with length equal to num_seqs"

    num_seqs, vocab_size = logits.shape

    block_size = _select_block_size(vocab_size)

    grid = (
        triton.cdiv(vocab_size, block_size),
        num_seqs,
    )

    _repetition_penalty_kernel[grid](
        logits,
        prompt_mask,
        output_mask,
        repetition_penalties,
        vocab_size,
        BLOCK_SIZE=block_size,
    )
    return None
