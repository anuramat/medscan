# medscan
### how to install:
1. docker build -t medscan .
2. docker container rm medscan # if already exists
3. docker create -p 5000:5000 --name medscan medscan
4. docker start medscan
