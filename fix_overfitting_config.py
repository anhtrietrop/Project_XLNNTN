# CẤU HÌNH ĐỂ GIẢM OVERFITTING
# Thay thế vào notebook của bạn

# =============================================================================
# 1. TĂNG DROPOUT (từ 0.5 → 0.6)
# =============================================================================
DROPOUT = 0.6  # Tăng regularization

# =============================================================================
# 2. GIẢM MODEL CAPACITY (từ 512 → 384)
# =============================================================================
HIDDEN_DIM = 384  # Giảm từ 512 để model không học vẹt

# =============================================================================
# 3. THÊM WEIGHT DECAY VÀO OPTIMIZER (L2 Regularization)
# =============================================================================
# Thay dòng:
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# Bằng:
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# =============================================================================
# 4. GIẢM TEACHER FORCING RATIO (từ 0.5 → 0.3)
# =============================================================================
# Trong hàm train_epoch(), thay:
# output = model(src, src_len, tgt, teacher_forcing_ratio=0.5)
# Bằng:
output = model(src, src_len, tgt, teacher_forcing_ratio=0.3)

# =============================================================================
# 5. GIẢM EARLY STOPPING PATIENCE (từ 10 → 5)
# =============================================================================
EARLY_STOPPING_PATIENCE = 5  # Dừng sớm hơn khi overfitting

# =============================================================================
# 6. THÊM LABEL SMOOTHING (Tùy chọn)
# =============================================================================
# Thay:
# criterion = nn.CrossEntropyLoss(ignore_index=fr_vocab.pad_idx)
# Bằng:
criterion = nn.CrossEntropyLoss(ignore_index=fr_vocab.pad_idx, label_smoothing=0.1)

# =============================================================================
# 7. THÊM GRADIENT CLIPPING MẠNH HƠN (từ 1.0 → 0.5)
# =============================================================================
CLIP = 0.5  # Giảm từ 1.0 để ổn định hơn

# =============================================================================
# TÓM TẮT CÁC THAY ĐỔI
# =============================================================================
"""
CÁC THAY ĐỔI ĐỂ GIẢM OVERFITTING:

1. ✓ DROPOUT: 0.5 → 0.6 (tăng 20%)
2. ✓ HIDDEN_DIM: 512 → 384 (giảm capacity 25%)
3. ✓ WEIGHT_DECAY: 0 → 1e-5 (thêm L2 regularization)
4. ✓ TEACHER_FORCING_RATIO: 0.5 → 0.3 (giảm 40%)
5. ✓ EARLY_STOPPING_PATIENCE: 10 → 5 (dừng sớm hơn)
6. ✓ LABEL_SMOOTHING: 0 → 0.1 (làm mềm labels)
7. ✓ GRADIENT_CLIP: 1.0 → 0.5 (ổn định hơn)

KỲ VỌNG SAU KHI THAY ĐỔI:
- Train loss sẽ cao hơn (~2.5-3.0 thay vì ~1.7)
- Val loss sẽ thấp hơn (~2.8-3.2 thay vì ~3.5)
- Khoảng cách Train-Val sẽ nhỏ hơn (~0.3-0.5 thay vì ~1.8)
- BLEU score có thể tăng nhẹ do generalize tốt hơn
"""
