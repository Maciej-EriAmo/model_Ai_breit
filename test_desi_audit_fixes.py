"""Testy po audycie 2026-09-10: split, size, labels, feats, mask."""

from __future__ import annotations

import numpy as np
import torch

from bridge import BridgeTransformer, SoftmaxTransformer
from desi_cutouts import label_from_demo_name
from image_batch import cutout_to_tokens, load_desi_train_val, patch_features


def test_label_exact_tokens_not_substring():
    assert label_from_demo_name("desi_leo_ra169.95_dec12.58_s128") == 1
    assert label_from_demo_name("desi_field_a_ra150.0_dec2.0_s128") == 0
    assert label_from_demo_name("desi_field_leonis_ra1_dec1_s128") == 0  # field wins
    assert label_from_demo_name("desi_cluster_abell85_ra10_dec-9_s128") == 1
    # substring trap: 'leo' inside unrelated without field
    assert label_from_demo_name("desi_chameleon_ra1_dec1_s128") == 0


def test_label_map_override():
    m = {"desi_weird_ra1_dec1_s128": "cluster", "weird": "cluster"}
    assert label_from_demo_name("desi_weird_ra1_dec1_s128", m) == 1


def test_patch_features_exactly_8():
    tile = np.random.randn(3, 16, 16).astype(np.float32)
    f = patch_features(tile, d_x=8)
    assert f.shape == (8,)
    assert np.isfinite(f).all()


def test_cutout_tokens_T_not_replaced_by_xy():
    img = np.random.rand(3, 64, 64).astype(np.float32)
    T = np.full((64, 64), 0.91, dtype=np.float32)
    c, tr = cutout_to_tokens(img, T, patch=16)
    assert c.shape[1] == 8
    assert tr.shape[1] == 6
    # slot 3 = T
    assert float(tr[:, 3].mean()) > 0.8


def test_color_to_T_differs_from_snr():
    from desi_cutouts import color_index_map, color_to_T, pack_photometry, snr_to_T, snr_map

    rng = np.random.default_rng(0)
    # sztuczne: pasmo g >> r w połowie kadru
    img = rng.random((3, 32, 32), dtype=np.float32) * 0.1
    img[0, :, :16] = 2.0
    img[1, :, :16] = 0.2
    photo = pack_photometry(img, None)
    assert photo["T_mode"] == "color_temperature"
    # T nie jest kopią SNR
    snr_t = snr_to_T(snr_map(img, None))
    assert float(np.mean(np.abs(photo["T"] - snr_t))) > 0.01


def test_mask_blocks_pad_in_bridge():
    B, N, dx, nd = 2, 8, 8, 6
    x = torch.randn(B, N, dx)
    tr = torch.randn(B, N, nd)
    mask = torch.ones(B, N)
    mask[:, N // 2 :] = 0.0  # druga połowa = pad
    m = BridgeTransformer(input_dim=dx, dim_tracer=nd, n_classes=2, num_layers=2)
    logits, w, dE = m(x, tr, mask=mask)
    assert logits.shape == (B, 2)
    assert w is not None
    # wagi na pad keys ~0 (po CLS: indeksy 1+N//2 ..)
    pad_keys = w[:, :, :, (1 + N // 2) :]
    assert float(pad_keys.detach().abs().mean()) < 1e-5


def test_softmax_mask_optional():
    B, N = 1, 4
    x = torch.randn(B, N, 8)
    tr = torch.zeros(B, N, 6)
    sm = SoftmaxTransformer(input_dim=8, dim_tracer=6, n_classes=2)
    logits, _, _ = sm(x, tr, mask=torch.ones(B, N))
    assert logits.shape == (B, 2)


def test_train_val_no_file_leak_if_cache():
    from pathlib import Path

    cache = Path("dumps/desi_cache")
    demos = list(cache.glob("demo_*_s128.npz"))
    if len(demos) < 4:
        return  # skip if no data
    train, val, meta = load_desi_train_val(
        demo_only=True,
        from_demo=False,
        size=128,
        augment=2,
        seed=0,
        val_frac=0.3,
    )
    assert meta["n_train_files"] >= 1
    assert meta["n_val_files"] >= 1
    leak = set(meta["train_files"]) & set(meta["val_files"])
    assert not leak, leak
    # val items == n_val_files (no augment)
    assert meta["n_val_items"] == meta["n_val_files"]
    assert meta["n_train_items"] == meta["n_train_files"] * (1 + 2)
    assert all(b.content.shape[1] == meta["seq_len"] for b in train + val)
    # mask covers all tokens when uniform size
    for b in train + val:
        assert float(b.mask.min()) == 1.0


if __name__ == "__main__":
    test_label_exact_tokens_not_substring()
    test_label_map_override()
    test_patch_features_exactly_8()
    test_cutout_tokens_T_not_replaced_by_xy()
    test_mask_blocks_pad_in_bridge()
    test_softmax_mask_optional()
    test_train_val_no_file_leak_if_cache()
    print("OK test_desi_audit_fixes")
