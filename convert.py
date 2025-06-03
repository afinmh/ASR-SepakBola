import os
from pydub import AudioSegment

def convert_audio_to_wav(folder):
    supported_formats = (".mp3", ".m4a", ".aac", ".flac", ".ogg", ".wma")
    for root, _, files in os.walk(folder):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_formats:
                input_path = os.path.join(root, file)
                output_path = os.path.join(root, os.path.splitext(file)[0] + ".wav")

                try:
                    print(f"🔄 Mengonversi {file} ke WAV...")
                    audio = AudioSegment.from_file(input_path)
                    audio.export(output_path, format="wav")
                    print(f"✅ Sukses: {output_path}")

                    # 🔥 Hapus file asli setelah berhasil konversi
                    os.remove(input_path)
                    print(f"🗑️ Dihapus: {input_path}")
                except Exception as e:
                    print(f"❌ Gagal konversi {file}: {e}")

if __name__ == "__main__":
    convert_audio_to_wav("data/train")
    convert_audio_to_wav("data/test")
