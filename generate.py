import argparse
import openai
from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor
from sentence_transformers import SentenceTransformer
import torch
from torch.nn import functional as F
import pandas as pd
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Generate')
    parser.add_argument('--api_key', type=str, default='sk-1234567890', help='OpenAI API key')
    parser.add_argument('--output_dir', type=str, default='.', help='output directory')
    parser.add_argument('--num_output', type=int, default=100, help='number of outputs')
    parser.add_argument('--motion_type', type=str, default='linear', help='Type of motion')
    parser.add_argument('--clip_similarity_thresh',
                        type=float,
                        default=0.9,
                        help='CLIP similarity threshold')
    parser.add_argument('--similarity_difference_thresh',
                        type=float,
                        default=0.2,
                        help='Difference between CLIP and BERT similarity threshold')
    args = parser.parse_args()
    return args

def load_clip_model():
    # Load the pre-trained CLIP model
    model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
    model = model.cuda()

    # Load the corresponding tokenizer
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    return model, tokenizer, processor


def load_bert_model():
    bert_model = SentenceTransformer('paraphrase-distilroberta-base-v1')

    return bert_model


def gpt_generate(prompt: str,
                 client: openai.OpenAI,):
    message = [
        {'role': 'user', 'content': prompt},
    ]

    response = client.chat.completions.create(
        messages = message,
        model = "gpt-3.5-turbo"
    )

    answer = response.choices[0].message.content.strip()
    answer = answer.split("\n")
    processed_answer = []
    for _, pair in enumerate(answer):
        processed_pair = string_to_list(pair)
        if len(processed_pair) == 2:
            processed_answer.append(processed_pair)

    return processed_answer


def string_to_list(s):
    # Remove the leading number and period if present, then strip any surrounding whitespace
    s = s.split('.', 1)[-1].strip()
    # Remove the trailing comma if present
    if s.endswith(','):
        s = s[:-1]
    if s.startswith('-'):
        s = s.replace('-', '')

    # Convert the string to a list
    try:
        result = eval(s)
        if isinstance(result, tuple):  # Check if the result is a tuple
            return list(result)  # Convert the tuple to a list
        elif isinstance(result, list):  # Check if the result is a list
            return result
    except SyntaxError:
        return []  # Return an empty list in case of a parsing error

    return []

def filter_pairs(pairs: list,
                 clip_model: CLIPModel,
                 clip_tokenizer: CLIPTokenizer,
                 bert_model: SentenceTransformer,
                 clip_similarity_thresh: float,
                 similarity_difference_thresh: float) -> list:
    filtered_pairs = {
        "prompt1": [],
        "prompt2": [],
        "clip_similarity": [],
        "bert_similarity": [],
        "difference": []
    }
    with torch.no_grad():
        for pair in pairs:
            bert_pairs_embeds = bert_model.encode(pair)
            bert_pairs_embeds = torch.from_numpy(bert_pairs_embeds).cuda()
            bert_pairs_embeds = F.normalize(bert_pairs_embeds, dim=1)
            tok = clip_tokenizer(pair,
                                return_tensors="pt",
                                padding=True,
                                truncation=True)

            for key in tok.keys():
                tok[key] = tok[key].cuda()
            clip_pairs_outputs = clip_model.text_model(**tok)
            clip_pairs_embeds = clip_pairs_outputs[1]
            clip_pairs_embeds = clip_model.text_projection(clip_pairs_embeds)
            clip_pairs_embeds = F.normalize(clip_pairs_embeds, dim=1).cuda()

            bert_similarity = torch.matmul(bert_pairs_embeds, bert_pairs_embeds.T)[1, 0]
            clip_similaritiy = torch.matmul(clip_pairs_embeds, clip_pairs_embeds.T)[1, 0]

            if clip_similaritiy > clip_similarity_thresh and \
                (clip_similaritiy - bert_similarity) > similarity_difference_thresh:
                filtered_pairs["prompt1"].append(pair[0])
                filtered_pairs["prompt2"].append(pair[1])
                filtered_pairs["clip_similarity"].append(clip_similaritiy.item())
                filtered_pairs["bert_similarity"].append(bert_similarity.item())
                filtered_pairs["difference"].append(clip_similaritiy.item() -\
                                                     bert_similarity.item())
    return filtered_pairs


def generate(args: argparse.Namespace):
    client = openai.OpenAI(api_key=args.api_key)
    if args.motion_type == 'linear':
        prompt = """Write down 41 additional pairs of prompts that\
                an embedding model with the following failure mode \
                might encode similarly, even though they would \
                correspond to different images if used as captions. Use the following format:
                ("prompt1", "prompt2"),
                ("prompt1", "prompt2"),
                You will be evaluated on how well you actually perform. \
                Your sentence structure and length can be creative; \
                extrapolate based on the failure mode you’ve summarized. Be both creative and cautious. \
                Please make the prompts complete sentences.
                

                Failure Mode:
                Linear Motion Contradiction: This type of contradiction occurs when \
                two scenarios describe movements that are directly opposite in direction \
                along a straight line, yet the subjects and environments remain identical \
                and no change farther or closer to the camera. \
                The core of this contradiction lies in the linear actions that counter each other,\
                such as upward movement versus downward movement, left movement versus right movement, and eastward movement versus westward movement.

                Example:
                ("A train moving forward down the tracks", "A train moving backward on the tracks"),
                ("A car driving east on a straight road", "A car driving west on a straight road"),

                Format:
                ("prompt1", "prompt2"),
                ("prompt1", "prompt2"),
                """

    elif args.motion_type == 'rotary':
        prompt = """Write down 41 additional pairs of prompts that\
                    an embedding model with the following failure mode \
                    might encode similarly, even though they would \
                    correspond to different images if used as captions. Use the following format:
                    ("prompt1", "prompt2"),
                    ("prompt1", "prompt2"),
                    You will be evaluated on how well you actually perform. \
                    Your sentence structure and length can be creative; \
                    extrapolate based on the failure mode you’ve summarized. Be both creative and cautious. \
                    Please make the prompts complete sentences.
                    
                    Failure Mode: 
                    Rotary Motion Contradiction: A contradiction occurs in scenarios of rotary motion \
                    when two described actions exhibit opposing rotational \
                    directions or intentions around the same axis or pivot point, \
                    with identical subjects and settings. This opposition is \
                    highlighted through pairs of rotational movements that are \
                    antithetical to each other, such as clockwise versus \
                    counterclockwise rotation or spinning versus unwinding.

                    Example:
                    ("A wheel spinning clockwise", "A wheel spinning counterclockwise"),
                    ("A person running around a track clockwise", "A person running around a track counterclockwise"),
                    ("A vinyl record rotating clockwise on a player", "A vinyl record rotating counterclockwise on a player"),
                    ("A ceiling fan turning clockwise to cool the room", "A ceiling fan turning counterclockwise to distribute heat"),
                    ("A dancer spinning on their left foot clockwise", "A dancer spinning on their left foot counterclockwise"),
                    ("A screw being tightened clockwise into wood", "A screw being loosened counterclockwise from wood"),

                    Format:
                    ("prompt1", "prompt2"),
                    ("prompt1", "prompt2"),
                    """

    elif args.motion_type == 'perspective':
        prompt = """Write down 41 additional pairs of prompts that\
                    an embedding model with the following failure mode \
                    might encode similarly, even though they would \
                    correspond to different images if used as captions. Use the following format:
                    ("prompt1", "prompt2"),
                    ("prompt1", "prompt2"),
                    You will be evaluated on how well you actually perform. \
                    Your sentence structure and length can be creative; \
                    extrapolate based on the failure mode you’ve summarized. Be both creative and cautious. \
                    Please make the prompts complete sentences.
                    
                    Failure Mode: 
                    Perspective Motion Contradiction: A contradiction in perspective motion arises when two scenarios depict actions \
                    or movements that fundamentally oppose each other in the context of the object's \
                    apparent size change due to its motion towards or away from the camera, yet the \
                    subjects and environments remain consistent. This opposition manifests in pairs of movements \
                    where one action results in the object appearing larger as it moves closer to the camera, \
                    and the other action makes it appear smaller as it moves away, despite identical starting conditions.

                    Example:
                    ("A persom walking closer to the camera, resulting in an increased projection size", "A person moving away from the camera, leading to a decreased projection size"),
                    ("A car approaching the camera, becoming increasingly larger in the frame", "A car receding from the camera, shrinking in the frame"),
                    ("A dog running towards the camera, appearing larger as it gets closer", "A dog running away from the camera, appearing smaller as it moves further away"),
                    ("A ball being thrown towards the camera, increasing in size in the frame", "A ball being thrown away from the camera, decreasing in size in the frame"),
                    ("A bird flying closer to the camera, growing in visual size", "A bird flying away from the camera, reducing in visual size"),
                    ("A cyclist pedaling towards the camera, becoming more prominent in the view", "A cyclist pedaling away from the camera, becoming less prominent in the view"),
                    ("A child walking closer to the camera, appearing taller", "A child walking away from the camera, appearing shorter"),
                    ("A plane landing towards the camera, looming larger", "A plane taking off away from the camera, shrinking in perspective"),
                    ("A train approaching the camera, filling more of the frame", "A train departing from the camera, filling less of the frame"),
                    ("A boat sailing towards the camera, becoming more detailed", "A boat sailing away from the camera, losing detail"),
                    ("A basketball player moving towards the camera for a dunk, appearing larger", "A basketball player moving away from the camera after a dunk, appearing smaller"),
                    
                    Format:
                    ("prompt1", "prompt2"),
                    ("prompt1", "prompt2"),
                    """
        
    elif args.motion_type == 'speed':
        prompt = """Write down 41 additional pairs of prompts that\
                    an embedding model with the following failure mode \
                    might encode similarly, even though they would \
                    correspond to different images if used as captions. Use the following format:
                    ("prompt1", "prompt2"),
                    ("prompt1", "prompt2"),
                    You will be evaluated on how well you actually perform. \
                    Your sentence structure and length can be creative; \
                    extrapolate based on the failure mode you’ve summarized. Be both creative and cautious. \
                    Please make the prompts complete sentences.

                    Failure Mode:
                    Speed Change Contradiction: the contradiction occurs when two scenarios \
                    describe the same subjects and environments but with movements that \
                    significantly differ in velocity, either accelerating or decelerating, \
                    yet are presented as occurring simultaneously. This type of contradiction \
                    is identified through comparisons of actions where one exhibits a significant \
                    increase in speed and the other a decrease, despite identical conditions.

                    Example:
                    ("A car accelerating on a highway", "A car decelerating on stretch of highway"),
                    ("A runner picking up pace in a race", "A runner slowing down during a race"),
                    ("A bicycle speeding up on a mountain trail", "A bicycle slowing down on a mountain trail"),
                    ("A horse galloping quickly across a field", "A horse trotting slowly across a field"),
                    ("A skateboarder accelerating in a skate park", "A skateboarder decelerating in a skate park"),
                    ("A dog running fast after a ball", "A dog walking leisurely towards a ball"),
                    ("A train gaining speed on the tracks", "A train coming to a halt on a tracks"),
                    ("A child sprinting joyfully in a park", "A child ambling slowly in a park"),
                    ("A bird soaring rapidly in the sky", "A bird gliding slowly in a part of the sky"),
                    ("A fish swimming swiftly in a stream", "A fish meandering lazily in a stream"),
                    ("A plane taking off rapidly from the runway", "A plane landing gently on a runway"),
                    ("A boat speeding through the water", "A boat drifting aimlessly on a water"),
                    ("A cheetah chasing prey at full speed", "A cheetah stalking prey at a cautious pace"),
                                        
                    Format:
                    ("prompt1", "prompt2"),
                    ("prompt1", "prompt2"),
                    """

    # prompt = """
    #         Failure Mode:
    #         Movement Contradiction: A contradiction exists between two scenarios \
    #         when their described actions or movements are diametrically opposed \
    #         in direction or intention but the subjects, and environments are exactly the same.\
    #         This opposition is manifest in pairs of \
    #         actions that are antithetical to each other, \
    #         such as entering versus exiting or ascending versus descending.

    #         Example:
    #         ("A person walking towards a building", "A person walking away from a building"),
    #         ("A bird flying upwards towards the sky", "A bird diving down towards the ground"),
    #     """
    final_contradicting_pairs = {
        "prompt1": [],
        "prompt2": [],
        "clip_similarity": [],
        "bert_similarity": [],
        "difference": []
    }

    clip_model, clip_tokenizer, _ = load_clip_model()
    bert_model = load_bert_model()
    run_count = 0
    while len(final_contradicting_pairs['prompt1']) < args.num_output:
        run_count += 1
        print(f"Run {run_count}...")
        contradicting_pairs = gpt_generate(prompt, client)
        filtered_pairs = filter_pairs(contradicting_pairs,
                                    clip_model,
                                    clip_tokenizer,
                                    bert_model,
                                    args.clip_similarity_thresh,
                                    args.similarity_difference_thresh)
        final_contradicting_pairs["prompt1"].extend(filtered_pairs["prompt1"])
        final_contradicting_pairs["prompt2"].extend(filtered_pairs["prompt2"])
        final_contradicting_pairs["clip_similarity"].extend(filtered_pairs["clip_similarity"])
        final_contradicting_pairs["bert_similarity"].extend(filtered_pairs["bert_similarity"])
        final_contradicting_pairs["difference"].extend(filtered_pairs["difference"])
        print(f"current length: {len(final_contradicting_pairs['prompt1'])}")
        time.sleep(5)
    return final_contradicting_pairs

def main():
    args = parse_args()
    final_contradicting_pairs = generate(args)
    df = pd.DataFrame(final_contradicting_pairs)
    df.to_csv(f"{args.output_dir}/contradicting_{args.motion_type}_pairs.csv", index=False)


if __name__ == '__main__':
    main()
