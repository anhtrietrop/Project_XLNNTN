# Neural Machine Translation: English-French (LSTM Encoder-Decoder)

## Đồ án Xử lý Ngôn ngữ Tự nhiên
**Học kỳ:** HK1 / 2025-2026

### Mô tả dự án
Triển khai từ đầu mô hình **Encoder-Decoder LSTM** để dịch máy từ tiếng Anh sang tiếng Pháp (Neural Machine Translation - NMT).

### Kiến trúc mô hình
- **Encoder:** 2-layer LSTM với embedding dimension 256, hidden size 384
- **Decoder:** 2-layer LSTM với greedy decoding
- **Dataset:** Multi30K (29,000 cặp câu train, 1,014 val, 1,000 test)
- **Vocabulary:** 10,000 từ phổ biến nhất mỗi ngôn ngữ

### Tham số mô hình
| Tham số | Giá trị |
|---------|---------|
| Embedding dimension | 256 |
| Hidden size | 384 |
| Số layer LSTM | 2 |
| Dropout | 0.6 |
| Teacher forcing ratio | 0.3 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Weight decay | 1e-5 |
| Label smoothing | 0.1 |
| Gradient clipping | 0.5 |

### Kết quả
- **BLEU Score:** ~25-35% trên tập test
- **Total parameters:** ~17.5M trainable parameters

### Cài đặt
```bash
# Tạo virtual environment
python -m venv .venv

# Activate environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Cài đặt dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torchtext spacy tqdm nltk datasets scikit-learn matplotlib seaborn pandas

# Tải spaCy models
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

### Dataset
Tải Multi30K dataset và đặt vào thư mục `data/`:
- `train.en.gz`, `train.fr.gz`
- `val.en.gz`, `val.fr.gz`
- `test_2016_flickr.en.gz`, `test_2016_flickr.fr.gz`

Hoặc download từ: [Multi30K GitHub](https://github.com/multi30k/dataset/tree/master/data/task1/raw)

### Sử dụng
1. Mở notebook `NMT_EnglishFrench_LSTM.ipynb`
2. Chạy các cell theo thứ tự từ đầu đến cuối
3. Model checkpoint sẽ được lưu tại `best_model.pth`

### Cấu trúc dự án
```
Project/
├── NMT_EnglishFrench_LSTM.ipynb    # Notebook chính
├── DO_AN_XLNNTN_FULL.txt           # Đề bài đồ án
├── data/                            # Dataset (không upload)
├── best_model.pth                   # Model checkpoint (không upload)
├── en_vocab.pkl, fr_vocab.pkl      # Vocabularies (không upload)
├── training_history.pkl            # Training curves (không upload)
└── README.md                        # File này
```

### Yêu cầu hệ thống
- **GPU:** NVIDIA GPU với CUDA 11.8+ (khuyến nghị)
- **RAM:** 8GB+ (16GB khuyến nghị)
- **Python:** 3.8+
- **PyTorch:** 2.7.1+cu118

### Tác giả
- **Sinh viên:** [Họ tên]
- **MSSV:** [Mã số sinh viên]

### License
MIT License
