
# 1) зависимости
pip install -r requirements.txt

# 2) простой пайплайн
python train_simple.py \
  --arch unetpp \
  --encoder resnet34 \
  --image-size 384 \
  --batch-size 4 \
  --epochs 28 \
  --lr 2e-4

