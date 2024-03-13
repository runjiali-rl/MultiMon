import argparse
import pandas as pd
import os
from diffusers.utils import export_to_gif
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--model_name', type=str, default='anime_lcm', help='model name for text2video')
    parser.add_argument('--prompts_path',
                        type=str,
                        default='similar_from_SNLI_top150_do_steer_True.csv',
                        help='input prompts')
    parser.add_argument('--motion_type', type=str, default=None, help='motion type')
    parser.add_argument('--output_dir', type=str, default='.', help='output image')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # Load prompts
    prompts = pd.read_csv(args.prompts_path)
    if args.model_name == 'anime_lcm':
        from models.anime_lcm import animate_lcm_generate
    elif args.model_name == 'modelscope':
        from models.modelscope import modelscope_generate
    elif args.model_name == 'zeroscope':
        from models.zeroscope import zeroscope_generate

    save_dir = os.path.join(args.output_dir, args.model_name, args.motion_type)
    os.makedirs(save_dir, exist_ok=True)
    print(prompts.head())
    for i in tqdm(range(1, len(prompts))):
        prompt = prompts.iloc[i]
        prompt_1 = prompt['prompt1']
        prompt_2 = prompt['prompt2']

        # Generate video
        if args.model_name == 'anime_lcm':
            test_output_1 = animate_lcm_generate(prompt_1)
            test_output_2 = animate_lcm_generate(prompt_2)
        elif args.model_name == 'modelscope':
            test_output_1 = modelscope_generate(prompt_1)
            test_output_2 = modelscope_generate(prompt_2)
        elif args.model_name == 'zeroscope':
            test_output_1 = zeroscope_generate(prompt_1)
            test_output_2 = zeroscope_generate(prompt_2)
        else:
            raise ValueError('Invalid model name')
        # Save video
        os.makedirs(os.path.join(save_dir,
                                 str(i)), exist_ok=True)
        save_path_1 = os.path.join(save_dir,
                                   str(i),
                                   f'{prompt_1.replace(" ", "_").replace(".", "")}.gif')
        save_path_2 = os.path.join(save_dir,
                                   str(i),
                                   f'{prompt_2.replace(" ", "_").replace(".", "")}.gif')
        export_to_gif(test_output_1, save_path_1)
        export_to_gif(test_output_2, save_path_2)


if __name__ == '__main__':
    main()
