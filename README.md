# Chihuahua

Command to extract audio from video:
```
ffmpeg -i files/dogs1.mp4 files/dogs1.wav
```


Command to combine audio and video:
```
ffmpeg -i files/dogs1-output.mp4 -i files/dogs1.wav -c:v copy -c:a aac -shortest files/dogs1-final.mp4
```