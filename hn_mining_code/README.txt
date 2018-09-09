Usage :-

python get_hardNegative.py --videoPath train_detection_videos/video${1} \
--detectionsFilePath train_detection_outputs/video${1}.txt \
--validScore 0.8 \
--alpha 5 \
--validSimScore 0.5 \
--iouThr 0.2 \
--beta 100

Note :-

Make sure the videoPath and detectionsFilePath are relative paths from where the get_hardNegative.py file is located and NOT absolute paths