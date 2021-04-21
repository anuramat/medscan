#!/bin/sh
docker run -it -p 5000:5000 --rm --mount "type=bind,source=$(pwd)/app/,target=/app/" medscan bash
