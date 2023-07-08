import torch
import os
import sys
from gooey import Gooey, GooeyParser
import cv2
from PIL import Image
import insightface
import imageio.v3 as iio
import warnings
from gooey import Gooey

warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# Set environment variable
os.environ["OMP_NUM_THREADS"] = "1"

# Initialize global variables
FACE_ANALYSER = None
FACE_SWAPPER = None
ONNX_PROVIDERS = ["CUDAExecutionProvider"]


def get_face_swapper():
    global FACE_SWAPPER
    if FACE_SWAPPER is None:
        model_path = "models/inswapper_128.onnx"
        FACE_SWAPPER = insightface.model_zoo.get_model(
            model_path, download=False, download_zip=False, providers=ONNX_PROVIDERS
        )
    return FACE_SWAPPER


def get_face_analyser():
    global FACE_ANALYSER
    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(name="buffalo_l", providers=ONNX_PROVIDERS)
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_ANALYSER


def parse_arguments():
    # python groop.py -t rune.jpg -i giphy.gif -o ooo.gif
    parser = GooeyParser(description="groop")
    parser.add_argument("-t", "--target", type=str, required=True, help="target face", widget="FileChooser")
    parser.add_argument("-i", "--input", type=str, required=True, help="input gif (can be a url)", widget="FileChooser")
    parser.add_argument("-o", "--output", type=str, required=True, help="output filename", widget="FileSaver")
    return parser.parse_args()


def process_gif(input_path, output_path, target_img):
    sys.stdout = open(os.devnull, "w")
    out_imgs = []

    tgt_img = cv2.imread(target_img)
    tgt_faces = sorted(get_face_analyser().get(tgt_img), key=lambda x: x.bbox[0])
    x = iio.immeta(input_path)
    duration = x["duration"]
    loop = x["loop"]
    gif = cv2.VideoCapture(input_path)

    while True:
        ret, frame = gif.read()
        if not ret:
            break
        out_faces = sorted(get_face_analyser().get(frame), key=lambda x: x.bbox[0])
        idx = 0
        for out_face in out_faces:
            frame = get_face_swapper().get(frame, out_face, tgt_faces[idx % len(tgt_faces)], paste_back=True)
            idx += 1
        out_imgs.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    path = output_path
    if not path.lower().endswith(".gif"):
        path = path + ".gif"

    out_imgs[0].save(path, save_all=True, append_images=out_imgs[1:], optimize=True, duration=duration, loop=loop)

    sys.stdout = sys.__stdout__


@Gooey
def main():
    # Parse command line arguments
    params = parse_arguments()

    # Process GIF
    print(f"Replacing face in {params.input} with {params.target} face...")
    process_gif(params.input, params.output, params.target)
    print(f"New gif created {params.output}")

    # Cleanup
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
