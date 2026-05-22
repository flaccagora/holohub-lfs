  
  ./holohub run-container tracks2endo4d

  ffmpeg -i /workspace/holohub/sample_video.mp4 -pix_fmt rgb24 -f rawvideo - \
    | python3 /workspace/holohub/utilities/convert_video_to_gxf_entities.py \
        --width 1280 \
        --height 720 \
        --channels 3 \
        --framerate 30 \
        --basename sample_video \
        --directory /workspace/holohub/data/tracks2endo4d/video

modifica entry in congig.yaml:

  sed -i 's/basename: ".*"/basename: "sample_video"/' \
    /workspace/holohub/applications/tracks2endo4d/config.yaml

  ./holohub run tracks2endo4d --language python --local --configure-args="-DCONVERT_ENGINE=OFF"


Per visualization

    ./holohub run tracks2endo4d \
        --configure-args="-DCONVERT_ENGINE=OFF" \
        --run-args="--viz-2d"
    