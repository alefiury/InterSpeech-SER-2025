# XEUS
if [ ! -d "ssl_checkpoints/XEUS" ]; then
    mkdir -p ssl_checkpoints/XEUS
fi
cd ssl_checkpoints/XEUS
wget https://huggingface.co/espnet/xeus/resolve/main/model/config.yaml
wget https://huggingface.co/espnet/xeus/resolve/main/model/xeus_checkpoint_new.pth
cd ../../
