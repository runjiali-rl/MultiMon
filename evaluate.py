from models.perception_models import Owlv2
from PIL import Image
import argparse
import os
import cv2
from tqdm import tqdm
import numpy as np
import torch
import spacy
import pickle
import os


FAILURE_TYPE = [
    "linear",
    "rotary",
    "speed",
    "perspective",
]

nlp = spacy.load("en_core_web_sm")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir",
                        type=str,
                        default="/homes/55/runjia/scratch/gen_video_results")
    parser.add_argument("--detector",
                        type=str,
                        default="owlv2")
    parser.add_argument("--visualize",
                        type=bool,
                        default=False)
    parser.add_argument("--failure_type",
                        type=str,
                        default="linear")
    return parser.parse_args()


def extract_nouns(text):
    # Process the text with spaCy
    doc = nlp(text)
    # Extract nouns from the text
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    nouns = [noun.lower() for noun in nouns if "ing" not in noun]
    return nouns



def extract_frames_from_video(video_path):
    """
    Extracts frames from a video file and returns them as a list of PIL Image objects.

    Parameters:
    - video_path: str. Path to the video file.

    Returns:
    - List of PIL.Image objects, each representing a frame from the video.
    """
    frames = []
    # Use OpenCV to capture the video
    cap = cv2.VideoCapture(video_path)

    # Check if the video has been opened successfully
    if not cap.isOpened():
        print("Failed to open the video")
        return frames

    # Read frames from the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if no more frames or error

        # Convert the frame from BGR to RGB color space
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert the OpenCV image (numpy array) to PIL Image and append to the list
        frames.append(Image.fromarray(frame_rgb))

    # Release the video capture object
    cap.release()
    return frames


def process_text(object_text):
    object_text = object_text.replace(".mp4", "")
    object_text = object_text.replace("_", " ")
    object_nouns = extract_nouns(object_text)
    words = object_text.split(" ")
    words = [word.lower() for word in words]
    # if the second and first nouns are next to each other, combine them
    if len(object_nouns) == 1:
        return object_nouns
    if object_nouns[0] in words and object_nouns[1] in words:
        if words.index(object_nouns[0]) + 1 == words.index(object_nouns[1]):
            object_nouns = [object_nouns[0] + " " + object_nouns[1]]
    return object_nouns



def judge_correctness(key_features_list: list,
                      failure_type: str,
                      time_steps_list: list,):
    """
    Judge the correctness of the detected boxes based on the failure type.
    args:
        key_features_list: list of list of key features, shape: (2, num_frames, 4)
        failure_type: str, the failure type
        time_steps_list: list of list of time steps
    """
    for key_features in key_features_list:
        if len(key_features) < 2:
            return False, False
    if failure_type == "linear":
        slope_list = []
        for key_features in key_features_list:
            start_x = (key_features[0][0] + key_features[0][2]) / 2
            start_y = (key_features[0][1] + key_features[0][3]) / 2
            end_x = (key_features[-1][0] + key_features[-1][2]) / 2
            end_y = (key_features[-1][1] + key_features[-1][3]) / 2

            # calculate the slope
            if end_x - start_x == 0:
                slope = float("inf")
            else:
                slope = (end_y - start_y) / (end_x - start_x)
            slope_list.append(slope)
        # if the two slopes have negative cosine similarity, then the detection is correct
        if np.dot(slope_list[0], slope_list[1]) < 0:
            return True, True
        else:
            return False, True
    elif failure_type == "speed":
        speed_list = []
        for idx, key_features in enumerate(key_features_list):
            start_x = (key_features[0][0] + key_features[0][2]) / 2
            start_y = (key_features[0][1] + key_features[0][3]) / 2
            end_x = (key_features[-1][0] + key_features[-1][2]) / 2
            end_y = (key_features[-1][1] + key_features[-1][3]) / 2
            speed = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)/\
                (time_steps_list[idx][-1] - time_steps_list[idx][0])
            speed_list.append(speed)
        if speed_list[0] > speed_list[1]:
            return True, True
        else:
            return False, True
    elif failure_type == "perspective":
        # the first bounding box should be larger as the object is closer
        # the second bounding box should be smaller as the object is further

        for idx, key_features in enumerate(key_features_list):
            start_bbox_size = (key_features[0][2] - key_features[0][0]) * \
                (key_features[0][3] - key_features[0][1])
            end_bbox_size = (key_features[-1][2] - key_features[-1][0]) * \
                (key_features[-1][3] - key_features[-1][1])
            if idx == 0:
                if start_bbox_size > end_bbox_size:
                    return True, True
                else:
                    return False, True
            else:
                if start_bbox_size < end_bbox_size:
                    return True, True
                else:
                    return False, True
        raise NotImplementedError(f"Failure type {failure_type} is not implemented yet.")



def evaluate_model(model_dir: str,
                   detector: Owlv2,
                   failure_type: str,
                   visualize: bool = False):
    assert failure_type in FAILURE_TYPE, f"Invalid failure type: {failure_type}"
    failure_results = {
        "idx": [],
        "correct_ratio": None,
        "exist_ratio": None,
    }
    failure_dir = os.path.join(model_dir, failure_type)
    total_num = 0
    exist_num = 0
    correct_num = 0
    for failure_file in tqdm(os.listdir(failure_dir)):
        failure_file_dir = os.path.join(failure_dir, failure_file)
        failure_pairs = os.listdir(failure_file_dir)
        key_features_list = []
        time_steps_list = []
        for failure_pair in failure_pairs:
            if not failure_pair.endswith(".mp4"):
                continue
            failure_video_path = os.path.join(failure_file_dir, failure_pair)
            failure_video = extract_frames_from_video(failure_video_path)
            object_nouns = process_text(failure_pair)
            object_prompt = f"a photo of a {object_nouns[0]}"
            key_features = []
            time_steps = []
            for time_step, failure_frame in enumerate(failure_video):
                # clean cuda cache
                torch.cuda.empty_cache()
                frame_result = detector.detect(object_prompt, failure_frame, threshold=0.4)
                boxes = frame_result["boxes"]
                if len(boxes) > 0:
                    key_features.append(boxes[0].tolist())
                    time_steps.append(time_step)
                # plot the boxes on the frame
                if visualize:
                    failure_frame_np = np.array(failure_frame)
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        cv2.rectangle(failure_frame_np,
                                        (int(x1), int(y1)),
                                        (int(x2), int(y2)),
                                        (0, 255, 0),
                                        2)
                    # save the frame
                    failure_frame_to_save = Image.fromarray(failure_frame_np)
                    failure_frame_to_save.save("test.jpg")
            key_features_list.append(key_features)
            time_steps_list.append(time_steps)
        # judge the correctness of the key features
        is_correct, is_object_exist = judge_correctness(key_features_list,
                                       failure_type,
                                       time_steps_list)
        if is_object_exist:
            exist_num += 1
            if is_correct:
                correct_num += 1
        failure_results['idx'].append(is_correct)
        total_num += 1
    failure_results["correct_ratio"] = correct_num / exist_num
    failure_results["exist_ratio"] = exist_num / total_num
    return failure_results


def evaluate(eval_dir: str,
             detector: Owlv2,
             failure_type: str,
             visualize: bool = False):
    evaluation_results = {}
    for model_name in os.listdir(eval_dir):
        model_dir = os.path.join(eval_dir, model_name)
        model_results = evaluate_model(model_dir,
                                       detector, 
                                       failure_type,
                                       visualize)
        evaluation_results[model_name] = model_results
        print(f"Model {model_name} has exist ratio: {model_results['exist_ratio']}")
        print(f"Model {model_name} has correct ratio: {model_results['correct_ratio']}")

    return evaluation_results


def main():
    args = parse_args()
    save_path = f"./outputs/evaluation_results_{args.failure_type}_{args.detector}.pkl"
    if os.path.exists(save_path):
        evaluation_results = pickle.load(open(save_path, "rb"))
        for model_name, model_results in evaluation_results.items():
            print(f"Model {model_name} has exist ratio: {model_results['exist_ratio']}")
            print(f"Model {model_name} has correct ratio: {model_results['correct_ratio']}")
        return
    if args.detector == "owlv2":
        detector = Owlv2(cache_dir="/homes/55/runjia/scratch/diffusion_weights")
    evaluation_results = evaluate(args.eval_dir,
                                    detector,
                                    args.failure_type,
                                    args.visualize)
    # save the evaluation results
    with open(save_path, "wb") as f:
        pickle.dump(evaluation_results, f)

if __name__ == '__main__':
    main()
