# git clone https://github.com/bethgelab/stylize-datasets.git
# git clone https://github.com/rois-codh/kaokore.git
# cd kaokore; python3 download.py --dataset_version 1.0

# cd ..;

python gen_lbl_wise_imagenet_style_dataset.py

python stylize-datasets/stylize.py --content-dir 'kaokore_imagenet_style/status/train/commoner' --style-dir 'kaokore_imagenet_style/status/train/commoner' --num-styles 1 --content-size 0 --style-size 224 --output-dir 'class_styled_kaokore/commoner'
python stylize-datasets/stylize.py --content-dir 'kaokore_imagenet_style/status/train/noble' --style-dir 'kaokore_imagenet_style/status/train/noble' --num-styles 1 --content-size 0 --style-size 224 --output-dir 'class_styled_kaokore/noble'
python stylize-datasets/stylize.py --content-dir 'kaokore_imagenet_style/status/train/incarnation' --style-dir 'kaokore_imagenet_style/status/train/incarnation' --num-styles 1 --content-size 0 --style-size 224 --output-dir 'class_styled_kaokore/incarnation'
python stylize-datasets/stylize.py --content-dir 'kaokore_imagenet_style/status/train/warrior' --style-dir 'kaokore_imagenet_style/status/train/warrior' --num-styles 1 --content-size 0 --style-size 224 --output-dir 'class_styled_kaokore/warrior'

cd class_styled_kaokore; mkdir allinone
cp commoner/* allinone/
cp noble/* allinone/
cp incarnation/* allinone/
cp warrior/* allinone/
python rename_files.py