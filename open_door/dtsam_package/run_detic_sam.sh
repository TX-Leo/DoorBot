IMAGE_PATH='/media/datadisk10tb/leo/projects/realman-robot/images/lever2.webp'
CLASSES="lever handle"
DEVICE="cuda:0"
THRESHOLD=0.2
/media/datadisk10tb/leo/anaconda3/envs/rm/bin/python detic_sam.py -i "$IMAGE_PATH" -c "$CLASSES" -d "$DEVICE" -t "$THRESHOLD"