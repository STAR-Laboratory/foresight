import json
import os

import tqdm

from videosys.utils.utils import set_seed


def generate_func(pipeline, prompt_list, output_dir, loop: int = 5, kwargs: dict = {}):
    kwargs["verbose"] = False
    for prompt in tqdm.tqdm(prompt_list):
        for l in range(loop):
            # set_seed(l)
            set_seed(1024)
            video = pipeline.generate(
                prompt,
                resolution="720p",
                aspect_ratio="9:16",
                num_frames="2s",
                flow=5,
                **kwargs,
            ).video[0]
            # pipeline.save_video(video, os.path.join(output_dir, f"{prompt}-{l}.mp4"))
            pipeline.save_video(video, os.path.join(output_dir, f"{prompt}.mp4"))

def generate_func_opensora_plan(pipeline, prompt_list, output_dir, loop: int = 5, num_inference_steps = 50, kwargs: dict = {}):
    kwargs["verbose"] = False
    for prompt in tqdm.tqdm(prompt_list):
        for l in range(loop):
            # set_seed(l)
            set_seed(1024)
            video = pipeline.generate(
                prompt=prompt,
                guidance_scale=7.5,
                num_inference_steps=num_inference_steps,
                seed=1024,
            ).video[0]
            # pipeline.save_video(video, os.path.join(output_dir, f"{prompt}-{l}.mp4"))
            pipeline.save_video(video, os.path.join(output_dir, f"{prompt}.mp4"))

def generate_func_latte(pipeline, prompt_list, output_dir, loop: int = 5, num_inference_steps = 50, kwargs: dict = {}):
    kwargs["verbose"] = False
    for prompt in tqdm.tqdm(prompt_list):
        for l in range(loop):
            # set_seed(l)
            set_seed(1024)
            video = pipeline.generate(
                prompt=prompt,
                guidance_scale=7.5,
                num_inference_steps=num_inference_steps,
                seed=1024,
            ).video[0]
            # pipeline.save_video(video, os.path.join(output_dir, f"{prompt}-{l}.mp4"))
            pipeline.save_video(video, os.path.join(output_dir, f"{prompt[0:20]}.mp4"))

def generate_func_cogvideox(pipeline, prompt_list, output_dir, loop: int = 5, num_inference_steps = 50, kwargs: dict = {}):
    kwargs["verbose"] = False
    for prompt in tqdm.tqdm(prompt_list):
        for l in range(loop):
            # set_seed(l)
            set_seed(1024)
            video = pipeline.generate(
                prompt=prompt,
                guidance_scale=6,
                num_inference_steps=num_inference_steps,
                num_frames=17,
                seed=1024,
            ).video[0]
            # pipeline.save_video(video, os.path.join(output_dir, f"{prompt}-{l}.mp4"))
            pipeline.save_video(video, os.path.join(output_dir, f"{prompt[0:20]}.mp4"))


def read_prompt_list(prompt_list_path):
    with open(prompt_list_path, "r") as f:
        prompt_list = json.load(f)
    prompt_list = [prompt["prompt_en"] for prompt in prompt_list]
    return prompt_list
