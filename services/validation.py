import os


def validate_ext(file):
    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".mp4",
        ".avi",
        ".mov",
        ".mkv"
    }

    file_name = file.filename

    ext = os.path.splitext(file_name)[1].lower()

    if ext not in extensions:
        raise ValueError("lỗi định dạng ko hợp lệ")


