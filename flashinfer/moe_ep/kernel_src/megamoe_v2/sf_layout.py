# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Scale-factor atom layout for NVFP4 tcgen05 operands.

This is a hardware ABI, not a design choice: tcgen05 reads NVFP4 block scales
from a swizzled 512-byte atom, and anything that *writes* scales -- the input
quantizer, the FC1 epilogue -- has to place bytes where the MMA will look for
them.  Getting it wrong does not fault; it silently scales the wrong elements.

One atom covers 128 tokens x 4 K-banks of one E4M3 byte each::

    byte_in_atom(t, k) = (t % 32) * 16 + (t // 32) * 4 + k
                         0 <= t < 128,  0 <= k < 4

Atoms tile K-major within a token row-block, then advance along rows::

    atom_index(row_block, k_atom) = row_block * num_k_atoms + k_atom

The functions here are pure integer arithmetic over Python ints so the layout
can be tested exhaustively on CPU (see the bijection test); the device code
evaluates the identical expression on ``Int32``.
"""

from __future__ import annotations

from .types import SF_ATOM_BLOCKS, SF_ATOM_ROWS

# One atom: 128 token rows x 4 K-banks x 1 byte.
SF_ATOM_BYTES = SF_ATOM_ROWS * SF_ATOM_BLOCKS
# The write granularity: 4 K-banks of one token are 4 consecutive bytes, which
# is exactly one int32 store.  Everything below is expressed in those units
# because that is what both writers actually emit.
SF_ATOM_WORDS = SF_ATOM_BYTES // 4
# Token rows sharing an inner swizzle group.
_SWIZZLE_GROUP = 32


def byte_in_atom(token_in_atom: int, k_bank: int) -> int:
    """Byte offset of one (token, K-bank) cell inside its 512-byte atom."""
    if not 0 <= token_in_atom < SF_ATOM_ROWS:
        raise ValueError(
            f"token_in_atom ({token_in_atom}) must be in [0, {SF_ATOM_ROWS})"
        )
    if not 0 <= k_bank < SF_ATOM_BLOCKS:
        raise ValueError(f"k_bank ({k_bank}) must be in [0, {SF_ATOM_BLOCKS})")
    return (
        (token_in_atom % _SWIZZLE_GROUP) * 16
        + (token_in_atom // _SWIZZLE_GROUP) * 4
        + k_bank
    )


def word_offset(token_row: int, k_atom: int, *, num_k_atoms: int) -> int:
    """int32-slot offset for one token's 4-bank group in a scale buffer.

    ``token_row`` is the row index *within the buffer* -- callers must already
    have added the expert's padded segment start, and that start must be a
    multiple of :data:`SF_ATOM_ROWS` or a token's scales would straddle two
    atoms.  :func:`~.layout.sf_row_capacity` guarantees the padding.
    """
    if token_row < 0:
        raise ValueError(f"token_row ({token_row}) must be non-negative")
    if not 0 <= k_atom < num_k_atoms:
        raise ValueError(f"k_atom ({k_atom}) must be in [0, {num_k_atoms})")
    row_block, token_in_atom = divmod(token_row, SF_ATOM_ROWS)
    atom_index = row_block * num_k_atoms + k_atom
    # byte_in_atom(t, 0) is always 4-aligned, so dividing by 4 is exact and the
    # int32 slot spans the cell's 4 K-banks.
    return atom_index * SF_ATOM_WORDS + byte_in_atom(token_in_atom, 0) // 4


def num_k_atoms_for(k_elements: int, block: int) -> int:
    """K-atoms per token row for a K extent quantized in ``block``-sized groups."""
    blocks = -(-k_elements // block)
    return -(-blocks // SF_ATOM_BLOCKS)


def buffer_words(rows: int, *, num_k_atoms: int) -> int:
    """int32 slots a scale buffer needs for ``rows`` padded token rows."""
    row_blocks = -(-rows // SF_ATOM_ROWS)
    return row_blocks * num_k_atoms * SF_ATOM_WORDS
