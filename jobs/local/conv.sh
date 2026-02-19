cd ../../
mkdir -p logs
python3 -u python -m src.train \
    --dataset soybean --root ./data \
    --defoca --P 4 --ratio 0.25 --sigma 1.0 \
    --strategy contiguous --V 8 --epochs 200 \
    --arch resnet18 \
    >> logs/r18.log 2>&1

python3 -u python -m src.train \
    --dataset soybean --root ./data \
    --defoca --P 4 --ratio 0.25 --sigma 1.0 \
    --strategy contiguous --V 8 --epochs 200 \
    --arch resnet50 \
    >> logs/r50.log 2>&1

python3 -u python -m src.train \
    --dataset soybean --root ./data \
    --defoca --P 4 --ratio 0.25 --sigma 1.0 \
    --strategy contiguous --V 8 --epochs 200 \
    --arch gcvit_tiny.in1k \
    >> logs/gcvit.log 2>&1