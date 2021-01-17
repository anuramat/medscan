#!/bin/bash
source /home/admin/miniconda3/bin/activate medscan
uvicorn run fastapi_code:app
