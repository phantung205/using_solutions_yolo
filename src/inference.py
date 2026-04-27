import argparse
import cv2


def get_args():
    parser = argparse.ArgumentParser("infence model yolo for vehicel")
    parser.add_argument("--image_path")