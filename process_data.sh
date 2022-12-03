cd data/
unzip train.zip
python get_frame.py -i train/videos/ -o train/images/
python create_data_h.py -dir train -images images -l label.csv
python create_data_s.py -dir train -images images -l label.csv

cd ..
python val_augmentation.py