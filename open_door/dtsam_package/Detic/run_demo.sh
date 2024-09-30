#!/bin/bash

# Set the parameters
CONFIG_FILE="configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
INPUT_FILE="images/input/handle_rgb_image_demo3.png"
OUTPUT_FILE="images/output/handle_rgb_image_demo3.png"
VOCABULARY="custom"
CUSTOM_VOCABULARY="key hole" #cross bar #bar handle # black key hole #handle #socket #push plate
CONFIDENCE_THRESHOLD="0.1"
MODEL_WEIGHTS="models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"

# Run the command with the parameters
python demo.py \
    --config-file "$CONFIG_FILE" \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --vocabulary "$VOCABULARY" \
    --custom_vocabulary "$CUSTOM_VOCABULARY" \
    --confidence-threshold "$CONFIDENCE_THRESHOLD" \
    --opts MODEL.WEIGHTS "$MODEL_WEIGHTS"