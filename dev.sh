#!/bin/sh
docker run -it --rm --mount "type=bind,source=$(pwd)/app/,target=/app/" medscan bash
