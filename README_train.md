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
- Tải file train.zip để vào trong thư mục data\
- Tiền xử lý dữ liệu: Cắt các frame với 1 frame/s.
- Chia dữ liệu

```bash
cd data\
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

# Huấn luyện mô hình

## Mô hình h

```bash
CUDA_VISIBLE_DEVICES=1 python main_h.py
```

## Mô hình s

```bash
CUDA_VISIBLE_DEVICES=1 python main_s.py
```