#!/bin/bash
source /home/admin/miniconda3/bin/activate medscan
uvicorn app:app --host 0.0.0.0 --port 5000