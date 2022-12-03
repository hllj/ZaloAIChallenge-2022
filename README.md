# Cài đặt

## Cài đặt pytorch
```bash
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

## Cài đặt các library

```bash
pip install -r pip_env.txt
```

## Cài đặt dữ liệu

### Bước 1:
- Tải file train.zip để vào trong thư mục data/
- Tiền xử lý dữ liệu: Cắt các frame với 1 frame/s.
- Chia dữ liệu

```bash
cd data/
unzip train.zip
python get_frame.py -i train/videos/ -o train/images/
python create_data_h.py -dir train -images images -l label.csv
python create_data_s.py -dir train -images images -l label.csv
```

### Bước 2: Tạo data augmentation cho tập train.

```bash
cd ..
python val_augmentation.py
```

**Note**: ta có thể sử dụng bash file. 
```bash
bash process_data.sh
```

# Hướng tiếp cận

Team sử dụng 2 hướng tiếp cận chính:

- Augmentation tập validation offline và giữ nguyên nó trong quá trình huấn luyện mô hình (mô hình h). Mục tiêu chính là team nhận thấy tập validation được sinh ra gần với tập public test 1 và 2 nhất và cho ra kết quả tốt nhất.

- Augmentation cả tập train và validation online trong lúc huấn luyện (mô hình s). Mục tiêu chính là để lấy được mô hình tốt nhất trên nhiều không gian khác nhau.

# Huấn luyện mô hình

## Mô hình h

```bash
CUDA_VISIBLE_DEVICES=1 python main_h.py
```

- Output sẽ là ở **outputs/h/ckpts**.
- Ở đây team mình lựa chọn mô hình cho ra **val_acc** lớn nhất (được đánh giá là tốt nhất trên tập public 1 và 2).

## Mô hình s

```bash
CUDA_VISIBLE_DEVICES=1 python main_s.py
```

- Output sẽ là ở **outputs/s/ckpts**.
- Ở đây team mình lựa chọn mô hình last.ckpt.

**Note**: Kết quả huấn luyện trên các device khác nhau có thể sai khác. Tụi mình đã có set seed và để CUDA Benchmark để mỗi lần huấn luyện là giống nhau trên 1 device.

# Inference

```bash
cp -r outputs/h weights/
cp -r outputs/s weights/
```

Sửa lại file config ở trong configs/inference.yaml.

```yaml
checkpoint_s: <Path to s folder>
checkpoint_h: <Path to h folder>
videos_dir: private_test/videos/*

hydra:
  run:
    dir: /result/
```

Chạy ensemble  cả 2 mô hình để predict.

```bash
CUDA_VISIBLE_DEVICES=0 python predict.py
```