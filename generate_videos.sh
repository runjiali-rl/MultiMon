
source ~/anaconda3/etc/profile.d/conda.sh
conda activate sora


# # Generate videos for Text2Video_Zero
# python inference.py --model_name text2video_zero --prompts_path prompts/contradicting_linear_pairs.csv --output_dir /homes/55/runjia/scratch/gen_video_results --motion_type linear
# python inference.py --model_name text2video_zero --prompts_path prompts/contradicting_rotary_pairs.csv --output_dir /homes/55/runjia/scratch/gen_video_results --motion_type rotary
# python inference.py --model_name text2video_zero --prompts_path prompts/contradicting_speed_pairs.csv --output_dir /homes/55/runjia/scratch/gen_video_results --motion_type speed
# python inference.py --model_name text2video_zero --prompts_path prompts/contradicting_perspective_pairs.csv --output_dir /homes/55/runjia/scratch/gen_video_results --motion_type perspective


# # Generate videos for Animate LCM
# python inference.py --model_name animate_lcm --prompts_path prompts/contradicting_linear_pairs.csv --output_dir /homes/55/runjia/scratch/gen_video_results --motion_type linear
# python inference.py --model_name animate_lcm --prompts_path prompts/contradicting_rotary_pairs.csv --output_dir /homes/55/runjia/scratch/gen_video_results --motion_type rotary
# python inference.py --model_name animate_lcm --prompts_path prompts/contradicting_speed_pairs.csv --output_dir /homes/55/runjia/scratch/gen_video_results --motion_type speed
# python inference.py --model_name animate_lcm --prompts_path prompts/contradicting_perspective_pairs.csv --output_dir /homes/55/runjia/scratch/gen_video_results --motion_type perspective

# # Generate videos for ModelScope
# python inference.py --model_name modelscope --prompts_path prompts/contradicting_linear_pairs.csv --output_dir /homes/55/runjia/scratch/gen_video_results --motion_type linear
# python inference.py --model_name modelscope --prompts_path prompts/contradicting_rotary_pairs.csv --output_dir /homes/55/runjia/scratch/gen_video_results --motion_type rotary
# python inference.py --model_name modelscope --prompts_path prompts/contradicting_speed_pairs.csv --output_dir /homes/55/runjia/scratch/gen_video_results --motion_type speed
# python inference.py --model_name modelscope --prompts_path prompts/contradicting_perspective_pairs.csv --output_dir /homes/55/runjia/scratch/gen_video_results --motion_type perspective

# # Generate videos for Zeroscope
# python inference.py --model_name zeroscope --prompts_path prompts/contradicting_linear_pairs.csv --output_dir /homes/55/runjia/scratch/gen_video_results --motion_type linear
# python inference.py --model_name zeroscope --prompts_path prompts/contradicting_rotary_pairs.csv --output_dir /homes/55/runjia/scratch/gen_video_results --motion_type rotary
# python inference.py --model_name zeroscope --prompts_path prompts/contradicting_speed_pairs.csv --output_dir /homes/55/runjia/scratch/gen_video_results --motion_type speed
# python inference.py --model_name zeroscope --prompts_path prompts/contradicting_perspective_pairs.csv --output_dir /homes/55/runjia/scratch/gen_video_results --motion_type perspective



# Generate videos for AnimateDiff
python inference.py --model_name animate_diff --prompts_path prompts/contradicting_linear_pairs.csv --output_dir /homes/55/runjia/scratch/gen_video_results --motion_type linear
python inference.py --model_name animate_diff --prompts_path prompts/contradicting_rotary_pairs.csv --output_dir /homes/55/runjia/scratch/gen_video_results --motion_type rotary
python inference.py --model_name animate_diff --prompts_path prompts/contradicting_speed_pairs.csv --output_dir /homes/55/runjia/scratch/gen_video_results --motion_type speed
python inference.py --model_name animate_diff --prompts_path prompts/contradicting_perspective_pairs.csv --output_dir /homes/55/runjia/scratch/gen_video_results --motion_type perspective

