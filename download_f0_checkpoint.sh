# XEUS
if [ ! -d "ssl_checkpoints/RMVPE" ]; then
    mkdir -p ssl_checkpoints/RMVPE
fi
cd ssl_checkpoints/RMVPE
wget "https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip"
unzip rmvpe.zip
rm rmvpe.zip
mv model.pt rmvpe.pt
cd ../../
