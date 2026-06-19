# Vietnamese Mispronunciation Detection and Diagnosis

Mã nguồn huấn luyện và suy luận hệ thống phát hiện, chẩn đoán lỗi phát âm
tiếng Việt bằng kiến trúc Prompt-FiLM CTC.

Pipeline chính được đặt tại
[`notebook/mdd_go.ipynb`](notebook/mdd_go.ipynb).

## Chức năng

- Đọc waveform và chuỗi âm vị chuẩn `canonical`.
- Mã hóa chuỗi chuẩn thành prompt vector.
- Điều kiện hóa đặc trưng âm học bằng FiLM.
- Huấn luyện với CTC loss.
- Đánh giá bằng F1, DER, PER và điểm tổng hợp của cuộc thi.
- Tìm ngưỡng confidence trên tập dev.
- Sinh `results.csv` và `prediction.zip` cho tập private.

## Yêu cầu

- Python 3.10 trở lên.
- GPU CUDA được khuyến nghị cho huấn luyện.
- Kaggle Notebook hoặc Google Colab.
- Bộ dữ liệu MDD Challenge 2025 được cấp quyền riêng.

Các thư viện chính:

```bash
pip install torch transformers numpy librosa soundfile tqdm kagglehub
```

## Chuẩn bị dữ liệu

Repository không chứa dữ liệu âm thanh hoặc metadata của cuộc thi. Cần tải và
gắn dữ liệu riêng vào môi trường chạy.

Cấu trúc dữ liệu training:

```text
MDD-Challenge-2025-training-set/
├── audio_data/
│   └── train/
│       └── *.wav
└── metadata/
    └── train_phones.csv
```

`train_phones.csv` cần có các cột:

```text
id,path,canonical,transcript
```

Cấu trúc dữ liệu private test:

```text
MDD-Challenge-2025-private-test/
├── audio_data/
│   └── private_test/
│       └── *.wav
└── metadata/
    └── private_test_submission.csv
```

File private test cần có các cột:

```text
id,path,canonical
```

Không đưa dữ liệu, file transcript đánh giá, checkpoint hoặc submission vào
Git.

## Cách chạy trên Kaggle

1. Mở `notebook/mdd_go.ipynb` trên Kaggle.
2. Thêm training dataset và private test dataset vào phần Input.
3. Bật GPU trong `Settings > Accelerator`.
4. Chạy lần lượt các cell từ trên xuống.
5. Kiểm tra log sau cell tìm dữ liệu:

```text
TRAIN_CSV
TRAIN_ROOT
PRIVATE_CSV
PRIVATE_ROOT
```

6. Sau khi huấn luyện hoàn tất, chạy cell private inference.
7. Tải file `prediction.zip` trong thư mục làm việc.

Trên Kaggle, thư mục đầu ra mặc định là:

```text
/kaggle/working/prompt_film_mdd/
```

## Cách chạy trên Google Colab

1. Tải `notebook/mdd_go.ipynb` lên Colab.
2. Chọn GPU tại `Runtime > Change runtime type`.
3. Gắn dataset bằng KaggleHub hoặc đặt dữ liệu trong Google Drive.
4. Chạy lần lượt các cell từ trên xuống.
5. Xác nhận notebook tìm đúng `TRAIN_CSV` và `PRIVATE_CSV`.
6. Sau huấn luyện, các file quan trọng được sao chép vào Google Drive.

Thư mục làm việc mặc định:

```text
/content/prompt_film_mdd/
```

Thư mục kết quả trên Drive:

```text
/content/drive/MyDrive/mdd_prompt_film_outputs/
```

## Cấu hình huấn luyện

Cấu hình nằm trong class `CFG` của notebook. Các giá trị mặc định:

| Tham số | Giá trị |
|---|---:|
| Backbone | `nguyenvulebinh/wav2vec2-base-vietnamese-250h` |
| Epoch | 45 |
| Batch size | 8 |
| Learning rate | `8e-6` |
| Dev ratio | 0.10 |
| Prompt encoder layers | 2 |
| Acoustic adapter layers | 4 |
| Adapter kernel | 5 |
| Attention heads | 8 |
| Dropout | 0.15 |
| Seed | 3407 |

Nếu GPU thiếu bộ nhớ, giảm:

```python
cfg.batch_size = 2
cfg.infer_batch_size = 1
```

Có thể tăng `grad_accum_steps` để giữ effective batch size.

## Trình tự notebook

Notebook được chạy theo thứ tự:

1. Khởi tạo môi trường và seed.
2. Khai báo cấu hình.
3. Tìm dữ liệu, chia train/dev và xây vocab.
4. Khởi tạo metric và hàm alignment.
5. Tạo Dataset, DataLoader và collate function.
6. Khởi tạo Prompt-FiLM CTC.
7. Huấn luyện và đánh giá trên dev.
8. Lưu checkpoint tốt nhất.
9. Suy luận private test.
10. Tìm ngưỡng confidence trên dev và xuất các candidate.

Không dùng nhãn public hoặc private để chọn ngưỡng. Việc chọn cấu hình decoder
chỉ thực hiện trên dev split được tách từ training set.

## File đầu ra

Sau khi chạy đầy đủ, thư mục làm việc có thể chứa:

```text
prompt_film_mdd/
├── best_wer.pt
├── best_score.pt
├── config.json
├── history.json
├── vocab_mdd2025.json
├── dev_predictions_best_wer.csv
├── results.csv
├── prediction.zip
├── prompt_film_model_bundle.zip
└── decoder_candidates/
```

Ý nghĩa:

- `best_wer.pt`: checkpoint có dev WER thấp nhất.
- `best_score.pt`: checkpoint có dev score cao nhất.
- `history.json`: metric theo từng epoch.
- `results.csv`: dự đoán cho private test.
- `prediction.zip`: file nộp kết quả.
- `decoder_candidates/`: các kết quả raw, dev-tuned, balanced và strict.

## Định dạng submission

File `results.csv` phải có đúng ba cột:

```text
id,path,predict
```

File ZIP chỉ chứa một file:

```text
results.csv
```

Notebook đã kiểm tra số lượng dòng, thứ tự `id` và tên file bên trong ZIP trước
khi xuất.

## Chính sách repository

Các nội dung sau được chặn bởi `.gitignore`:

- Dataset và file WAV.
- Public/private metadata.
- Transcript dùng để đánh giá.
- Checkpoint và model weights.
- Prediction và submission.
- File nén.
- Tài liệu, slide, hình ảnh và sơ đồ.
- Credential, token và cấu hình môi trường cá nhân.

Trước khi commit, nên kiểm tra:

```bash
git status --short
git diff --cached --name-only
```

Không commit file nếu chưa xác định rõ nguồn gốc hoặc quyền công khai.
