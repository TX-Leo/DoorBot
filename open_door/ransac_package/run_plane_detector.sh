RGB='/media/datadisk10tb/leo/projects/realman-robot/open_door/images/image1/rgb.png'
D='/media/datadisk10tb/leo/projects/realman-robot/open_door/images/image1/d.png'
CFG="/media/datadisk10tb/leo/projects/realman-robot/open_door/images/image1/ransac/ransac_cfg.yaml"
CAMERA="/media/datadisk10tb/leo/projects/realman-robot/open_door/images/image1/ransac/cam_params.json"
V="True"
O='horizontal'

# Run the command with the parameters
# python plane_detector.py -rgb "$RGB" -d "$D" -cfg "$CFG" -camera "$CAMERA" -v "$V" -o "$O"
if [ "${V}" = "True" ]; then
  python plane_detector.py -rgb "${RGB}" -d "${D}" -cfg "${CFG}" -camera "${CAMERA}" -v -o "${O}"
else
  python plane_detector.py -rgb "${RGB}" -d "${D}" -cfg "${CFG}" -camera "${CAMERA}"  -o "${O}"
fi