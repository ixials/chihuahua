# Chihuahua

Command to combine audio and video:
```
ffmpeg -i files/dogs1-output.mp4 -i files/dogs1.wav -c:v copy -c:a aac -shortest files/dogs1-final.mp4
```