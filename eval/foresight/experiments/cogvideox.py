from utils import read_prompt_list, generate_func_cogvideox

from videosys import CogVideoXConfig, CogVideoXPABConfig, CogVideoXFORESIGHTConfig, VideoSysEngine

# ========== Baseline ============
def eval_base_vbench(prompt_list):
    config = CogVideoXConfig("THUDM/CogVideoX-2b", num_gpus=1)
    engine = VideoSysEngine(config)
    generate_func_cogvideox(engine, prompt_list, "./vbench_cogvideox_baseline_50steps", loop=1, num_inference_steps=50)

def eval_base_evalcrafter(prompt_list):
    config = CogVideoXConfig("THUDM/CogVideoX-2b", num_gpus=1)
    engine = VideoSysEngine(config)
    generate_func_cogvideox(engine, prompt_list, "./evalcrafter_cogvideox_baseline_50steps", loop=1, num_inference_steps=50)

def eval_base_ucf(prompt_list):
    config = CogVideoXConfig("THUDM/CogVideoX-2b", num_gpus=1)
    engine = VideoSysEngine(config)
    generate_func_cogvideox(engine, prompt_list, "./ucf_cogvideox_bseline_50steps", loop=1, num_inference_steps=50)

def eval_base_sora(prompt_list):
    config = CogVideoXConfig("THUDM/CogVideoX-2b", num_gpus=1)
    engine = VideoSysEngine(config)
    generate_func_cogvideox(engine, prompt_list, "./sora_cogvideox_baseline_50steps", loop=1, num_inference_steps=50)

# ========== PAB ============
def eval_pab_vbench(prompt_list):
    pab_config = CogVideoXPABConfig(
        spatial_range=2,
    )
    config = CogVideoXConfig("THUDM/CogVideoX-2b", enable_pab=True, pab_config=pab_config)
    engine = VideoSysEngine(config)
    generate_func_cogvideox(engine, prompt_list, "./vbench_cogvideox_pab2_50steps", loop=1, num_inference_steps=50)

def eval_pab_evalcrafter(prompt_list):
    pab_config = CogVideoXPABConfig(
        spatial_range=2,
    )
    config = CogVideoXConfig("THUDM/CogVideoX-2b", enable_pab=True, pab_config=pab_config)
    engine = VideoSysEngine(config)
    generate_func_cogvideox(engine, prompt_list, "./evalcrafter_cogvideox_pab2_50steps", loop=1, num_inference_steps=50)


def eval_pab_ucf(prompt_list):
    pab_config = CogVideoXPABConfig(
        spatial_range=2,
    )
    config = CogVideoXConfig("THUDM/CogVideoX-2b", enable_pab=True, pab_config=pab_config)
    engine = VideoSysEngine(config)
    generate_func_cogvideox(engine, prompt_list, "./ucf_cogvideox_pab2_50steps", loop=1, num_inference_steps=50)

def eval_pab_sora(prompt_list):
    pab_config = CogVideoXPABConfig(
        spatial_range=2,
    )
    config = CogVideoXConfig("THUDM/CogVideoX-2b", enable_pab=True, pab_config=pab_config)
    engine = VideoSysEngine(config)
    generate_func_cogvideox(engine, prompt_list, "./sora_cogvideox_pab2_50steps", loop=1, num_inference_steps=50)

# ========== Foresight ============

def eval_foresight_evalcrafter(prompt_list):
    foresight_config = CogVideoXFORESIGHTConfig(
        warmup=8,
        recalculate=2,
        threshold=0.5,
    )
    config = CogVideoXConfig("THUDM/CogVideoX-2b", enable_foresight=True, foresight_config=foresight_config)
    engine = VideoSysEngine(config)
    generate_func_cogvideox(engine, prompt_list, "./evalcrafter_cogvideox_foresight_8_2_0.5_50steps", loop=1, num_inference_steps=50)


def eval_foresight_ucf(prompt_list):
    foresight_config = CogVideoXFORESIGHTConfig(
        warmup=8,
        recalculate=2,
        threshold=0.5,
    )
    config = CogVideoXConfig("THUDM/CogVideoX-2b", enable_foresight=True, foresight_config=foresight_config)
    engine = VideoSysEngine(config)
    generate_func_cogvideox(engine, prompt_list, "./ucf_cogvideox_foresight_8_2_0.5_50steps", loop=1, num_inference_steps=50)

def eval_foresight_sora(prompt_list):
    foresight_config = CogVideoXFORESIGHTConfig(
        warmup=8,
        recalculate=2,
        threshold=0.5,
    )
    config = CogVideoXConfig("THUDM/CogVideoX-2b", enable_foresight=True, foresight_config=foresight_config)
    engine = VideoSysEngine(config)
    generate_func_cogvideox(engine, prompt_list, "./sora_cogvideox_foresight_8_2_0.5_50steps", loop=1, num_inference_steps=50)


def eval_foresight_vbench(prompt_list, warmup, recalculate, threshold):
    foresight_config = CogVideoXFORESIGHTConfig(
        warmup=warmup,
        recalculate=recalculate,
        threshold=threshold,
    )
    config = CogVideoXConfig("THUDM/CogVideoX-2b", enable_foresight=True, foresight_config=foresight_config)
    engine = VideoSysEngine(config)
    output_dir = f'./vbench_cogvideox_foresight_{warmup}_{recalculate}_{threshold}_50steps'
    generate_func_cogvideox(engine, prompt_list, output_dir, loop=1, num_inference_steps=50)


if __name__ == "__main__":
    prompt_list = read_prompt_list("vbench/VBench_full_info_50.json")
    
    #with open('../assets/texts/prompt_evalcrafter.txt', "r") as f:
    #   prompt_list = [line.strip() for line in f.readlines()]

    #with open('../assets/texts/prompt_ucf.txt', "r") as f:
    #   prompt_list = [line.strip() for line in f.readlines()]

    #with open('../assets/texts/t2v_sora.txt', "r") as f:
    #   prompt_list = [line.strip() for line in f.readlines()]

    warmup = [8]
    recalculate = [2]
    threshold = [0.5]
    for i, w in enumerate(warmup):
        for j, r in enumerate(recalculate):
            for k, t in enumerate(threshold):
                print(f'VBench Foresight Wamrup {w}, Recalculate {r}, Threshold {t} with 50 steps')
                eval_foresight_vbench(prompt_list, w, r, t)