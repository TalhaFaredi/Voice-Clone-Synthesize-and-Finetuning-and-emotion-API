from speechbrain.pretrained import SpeakerRecognition




verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)


file1 = "Ava-fearful.wav"
file2 = "Ava-fearful.wav"


score, prediction = verification.verify_files(file1, file2)

print("Similarity score:", score.item())
print("Same speaker?", "Yes" if prediction.item() == 1 else "No")


from moviepy import VideoFileClip
import os

def convert_mp4_to_mp3(mp4_path, mp3_output_path=None):
    if not mp4_path.endswith(".mp4"):
        raise ValueError("Input file must be an MP4 file.")

    if mp3_output_path is None:
        mp3_output_path = os.path.splitext(mp4_path)[0] + ".mp3"

    try:
        print(f"Converting {mp4_path} to {mp3_output_path}...")
        video = VideoFileClip(mp4_path)
        audio = video.audio
        audio.write_audiofile(mp3_output_path)
        print("Conversion successful.")
    except Exception as e:
        print(f"Error during conversion: {e}")

convert_mp4_to_mp3("banda.mp4")
# import moviepy.editor as mpy
# print(mpy.__file__)
# from moviepy.editor import VideoFileClip
# import os

# def convert_mp4_to_mp3(mp4_path, mp3_output_path=None):
#     if not mp4_path.endswith(".mp4"):
#         raise ValueError("Input file must be an MP4 file.")

#     if mp3_output_path is None:
#         mp3_output_path = os.path.splitext(mp4_path)[0] + ".mp3"

#     try:
#         print(f"Converting {mp4_path} to {mp3_output_path}...")
#         video = VideoFileClip(mp4_path)
#         audio = video.audio
#         audio.write_audiofile(mp3_output_path)
#         print("Conversion successful.")
#     except Exception as e:
#         print(f"Error during conversion: {e}")

# convert_mp4_to_mp3("banda.mp4")