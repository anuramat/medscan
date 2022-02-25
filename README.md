# MEDSCAN
In order to build the image, run the following commands from the root of the repo folder:
```bash
sudo docker build .
sudo docker run -d -p %TARGET_PORT%:5000 --name medscan_container --mount "type=bind,source=$(pwd)/app/,target=/app/" --restart always medscan
```
