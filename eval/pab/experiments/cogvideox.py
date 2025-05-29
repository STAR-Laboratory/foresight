from utils import read_prompt_list, generate_func_cogvideox

from videosys import CogVideoXConfig, CogVideoXPABConfig, CogVideoXFORESIGHTConfig, VideoSysEngine

# ========== Baseline ============
def eval_base_vbench(prompt_list):
    config = CogVideoXConfig("THUDM/CogVideoX-2b", num_gpus=1)
    engine = VideoSysEngine(config)
    generate_func_cogvideox(engine, prompt_list, "./samples/cogvideox/vbench_cogvideox_static_2_50steps", loop=1, num_inference_steps=50)

def eval_base_evalcrafter(prompt_list):
    config = CogVideoXConfig("THUDM/CogVideoX-2b", num_gpus=1)
    engine = VideoSysEngine(config)
    generate_func_cogvideox(engine, prompt_list, "./samples/cogvideox/evalcrafter_cogvideox_static_2_50steps", loop=1, num_inference_steps=50)

def eval_base_ucf(prompt_list):
    config = CogVideoXConfig("THUDM/CogVideoX-2b", num_gpus=1)
    engine = VideoSysEngine(config)
    generate_func_cogvideox(engine, prompt_list, "./samples/cogvideox/ucf_cogvideox_static_2_50steps", loop=1, num_inference_steps=50)

def eval_base_sora(prompt_list):
    config = CogVideoXConfig("THUDM/CogVideoX-2b", num_gpus=1)
    engine = VideoSysEngine(config)
    generate_func_cogvideox(engine, prompt_list, "./samples/cogvideox/sora_cogvideox_static_2_middle_layers_50steps", loop=1, num_inference_steps=50)

# ========== PAB ============
def eval_pab_vbench(prompt_list):
    pab_config = CogVideoXPABConfig(
        spatial_range=2,
    )
    config = CogVideoXConfig("THUDM/CogVideoX-2b", enable_pab=True, pab_config=pab_config)
    engine = VideoSysEngine(config)
    generate_func_cogvideox(engine, prompt_list, "./samples/cogvideox/vbench_cogvideox_pab246_50steps", loop=1, num_inference_steps=50)

def eval_pab_evalcrafter(prompt_list):
    pab_config = CogVideoXPABConfig(
        spatial_range=2,
    )
    config = CogVideoXConfig("THUDM/CogVideoX-2b", enable_pab=True, pab_config=pab_config)
    engine = VideoSysEngine(config)
    generate_func_cogvideox(engine, prompt_list, "./samples/cogvideox/evalcrafter_cogvideox_pab246_50steps", loop=1, num_inference_steps=50)


def eval_pab_ucf(prompt_list):
    pab_config = CogVideoXPABConfig(
        spatial_range=2,
    )
    config = CogVideoXConfig("THUDM/CogVideoX-2b", enable_pab=True, pab_config=pab_config)
    engine = VideoSysEngine(config)
    generate_func_cogvideox(engine, prompt_list, "./samples/cogvideox/ucf_cogvideox_pab246_50steps", loop=1, num_inference_steps=50)

def eval_pab_sora(prompt_list):
    pab_config = CogVideoXPABConfig(
        spatial_range=2,
    )
    config = CogVideoXConfig("THUDM/CogVideoX-2b", enable_pab=True, pab_config=pab_config)
    engine = VideoSysEngine(config)
    generate_func_cogvideox(engine, prompt_list, "./samples/cogvideox/sora_cogvideox_pab246_50steps", loop=1, num_inference_steps=50)

# ========== Foresight ============

def eval_foresight_evalcrafter(prompt_list):
    foresight_config = CogVideoXFORESIGHTConfig(
        warmup=8,
        recalculate=2,
        threshold=1,
    )
    config = CogVideoXConfig("THUDM/CogVideoX-2b", enable_foresight=True, foresight_config=foresight_config)
    engine = VideoSysEngine(config)
    generate_func_cogvideox(engine, prompt_list, "./samples/cogvideox/evalcrafter_cogvideox_foresight_warmup-8_recalculate-2_threshold-1_50steps", loop=1, num_inference_steps=50)


def eval_foresight_ucf(prompt_list):
    foresight_config = CogVideoXFORESIGHTConfig(
        warmup=8,
        recalculate=2,
        threshold=1,
    )
    config = CogVideoXConfig("THUDM/CogVideoX-2b", enable_foresight=True, foresight_config=foresight_config)
    engine = VideoSysEngine(config)
    generate_func_cogvideox(engine, prompt_list, "./samples/cogvideox/ucf_cogvideox_foresight_warmup-8_recalculate-2_threshold-1_50steps", loop=1, num_inference_steps=50)

def eval_foresight_sora(prompt_list):
    foresight_config = CogVideoXFORESIGHTConfig(
        warmup=8,
        recalculate=2,
        threshold=1,
    )
    config = CogVideoXConfig("THUDM/CogVideoX-2b", enable_foresight=True, foresight_config=foresight_config)
    engine = VideoSysEngine(config)
    generate_func_cogvideox(engine, prompt_list, "./samples/cogvideox/sora_cogvideox_foresight_warmup-8_recalculate-2_threshold-1_50steps", loop=1, num_inference_steps=50)


def eval_foresight_vbench(prompt_list, warmup, recalculate, threshold):
    foresight_config = CogVideoXFORESIGHTConfig(
        warmup=warmup,
        recalculate=recalculate,
        threshold=threshold,
    )
    config = CogVideoXConfig("THUDM/CogVideoX-2b", enable_foresight=True, foresight_config=foresight_config)
    engine = VideoSysEngine(config)
    output_dir = f'./samples/cogvideox/sora_cogvideox_foresight_warmup-{warmup}_recalculate-{recalculate}_threshold-{threshold}_50steps'
    generate_func_cogvideox(engine, prompt_list, output_dir, loop=1, num_inference_steps=50)


if __name__ == "__main__":
    #prompt_list = read_prompt_list("vbench/VBench_full_info_50.json")
    #print(f'VBench Static-2 50 steps')
    #eval_base_vbench(prompt_list)
    
    #with open('../../../assets/texts/prompt_evalcrafter.txt', "r") as f:
    #   prompt_list = [line.strip() for line in f.readlines()]

    #print(f'Evalcrafter Static-2 50 steps')
    #eval_base_evalcrafter(prompt_list)

    #with open('../../../assets/texts/prompt_ucf.txt', "r") as f:
    #   prompt_list = [line.strip() for line in f.readlines()]

    #print(f'UCF Static-2 50 steps')
    #eval_base_ucf(prompt_list)

    with open('../../../assets/texts/t2v_sora.txt', "r") as f:
       prompt_list = [line.strip() for line in f.readlines()]

    warmup = [3, 5, 10, 13, 15, 18, 20]
    recalculate = [2]
    threshold = [0.5]
    for i, w in enumerate(warmup):
        for j, r in enumerate(recalculate):
            for k, t in enumerate(threshold):
                print(f'Sora Foresight Wamrup {w}, Recalculate {r}, Threshold {t} with 50 steps')
                eval_foresight_vbench(prompt_list, w, r, t)
    

    #print(f'VBench PAB246 50 steps')
    #eval_pab_vbench(prompt_list)

    #with open('../../../assets/texts/prompt_evalcrafter.txt', "r") as f:
    #    prompt_list = [line.strip() for line in f.readlines()]
    
    #print(f'Evalcrafter Baseline 50 steps')
    #eval_base_evalcrafter(prompt_list)

    #print(f'Evalcrafter PAB246 50 steps')
    #eval_pab_evalcrafter(prompt_list)

    #with open('../../../assets/texts/prompt_ucf.txt', "r") as f:
    #    prompt_list = [line.strip() for line in f.readlines()]

    #print(f'UCF Baseline 50 steps')
    #eval_base_ucf(prompt_list)

    #print(f'UCF PAB246 50 steps')
    #eval_pab_ucf(prompt_list)

    #with open('../../../assets/texts/t2v_sora.txt', "r") as f:
    #    prompt_list = [line.strip() for line in f.readlines()]

    #print(f'Sora PAB246 50 steps')
    #eval_base_sora(prompt_list)

    #print(f'Sora PAB246 50 steps')
    #eval_pab_sora(prompt_list)

