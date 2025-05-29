from utils import generate_func, read_prompt_list, generate_func_latte

from videosys import LatteConfig, LattePABConfig, LatteFORESIGHTConfig, VideoSysEngine

# ========== Baseline ============

def eval_base_evalcrafter(prompt_list):
    config = LatteConfig()
    engine = VideoSysEngine(config)
    generate_func_latte(engine, prompt_list, "./samples/latte/evalcrafter_latte_baseline_50steps", loop=1, num_inference_steps=50)

def eval_base_ucf(prompt_list):
    config = LatteConfig()
    engine = VideoSysEngine(config)
    generate_func_latte(engine, prompt_list, "./samples/latte/ucf_latte_baseline_50steps", loop=1, num_inference_steps=50)

def eval_base_sora(prompt_list):
    config = LatteConfig()
    engine = VideoSysEngine(config)
    generate_func_latte(engine, prompt_list, "./samples/latte/sora_latte_baseline_50steps", loop=1, num_inference_steps=50)

# ========== PAB ============

def eval_pab_evalcrafter(prompt_list):
    pab_config = LattePABConfig(
        spatial_range=2,
        temporal_range=4,
        cross_range=6,
    )
    config = LatteConfig(enable_pab=True, pab_config=pab_config)
    engine = VideoSysEngine(config)
    generate_func_latte(engine, prompt_list, "./samples/latte/evalcrafter_latte_pab246_50steps", loop=1, num_inference_steps=50)


def eval_pab_ucf(prompt_list):
    pab_config = LattePABConfig(
        spatial_range=2,
        temporal_range=4,
        cross_range=6,
    )
    config = LatteConfig(enable_pab=True, pab_config=pab_config)
    engine = VideoSysEngine(config)
    generate_func_latte(engine, prompt_list, "./samples/latte/ucf_latte_pab246_50steps", loop=1, num_inference_steps=50)

def eval_pab_sora(prompt_list):
    pab_config = LattePABConfig(
        spatial_range=2,
        temporal_range=4,
        cross_range=6,
    )
    config = LatteConfig(enable_pab=True, pab_config=pab_config)
    engine = VideoSysEngine(config)
    generate_func_latte(engine, prompt_list, "./samples/latte/sora_latte_pab246_50steps", loop=1, num_inference_steps=50)

# ========== Foresight ============

def eval_foresight_evalcrafter(prompt_list):
    foresight_config = LatteFORESIGHTConfig(
        warmup=8,
        recalculate=2,
        threshold=1,
    )
    config = LatteConfig(enable_foresight=True, foresight_config=foresight_config)
    engine = VideoSysEngine(config)
    generate_func_latte(engine, prompt_list, "./samples/latte/evalcrafter_latte_foresight_warmup-8_recalculate-2_threshold-1_50steps", loop=1, num_inference_steps=50)


def eval_foresight_ucf(prompt_list):
    foresight_config = LatteFORESIGHTConfig(
        warmup=8,
        recalculate=2,
        threshold=1,
    )
    config = LatteConfig(enable_foresight=True, foresight_config=foresight_config)
    engine = VideoSysEngine(config)
    generate_func_latte(engine, prompt_list, "./samples/latte/ucf_latte_foresight_warmup-8_recalculate-2_threshold-1_50steps", loop=1, num_inference_steps=50)

def eval_foresight_sora(prompt_list):
    foresight_config = LatteFORESIGHTConfig(
        warmup=8,
        recalculate=2,
        threshold=1,
    )
    config = LatteConfig(enable_foresight=True, foresight_config=foresight_config)
    engine = VideoSysEngine(config)
    generate_func_latte(engine, prompt_list, "./samples/latte/sora_latte_foresight_warmup-8_recalculate-2_threshold-1_50steps", loop=1, num_inference_steps=50)


def eval_foresight_vbench(prompt_list, warmup, recalculate, threshold):
    foresight_config = LatteFORESIGHTConfig(
        warmup=warmup,
        recalculate=recalculate,
        threshold=threshold,
    )
    config = LatteConfig(enable_foresight=True, foresight_config=foresight_config)
    engine = VideoSysEngine(config)
    output_dir = f'./samples/latte/sora_latte_foresight_warmup-{warmup}_recalculate-{recalculate}_threshold-{threshold}_50steps'
    generate_func_latte(engine, prompt_list, output_dir, loop=1, num_inference_steps=50)


if __name__ == "__main__":
    #prompt_list = read_prompt_list("vbench/VBench_full_info_50.json")
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
    
    #with open('../../../assets/texts/prompt_evalcrafter.txt', "r") as f:
    '''
    with open('../../../assets/texts/t2v_sora.txt', "r") as f:
        prompt_list = [line.strip() for line in f.readlines()]
    print(f'Sora Baseline 50 steps')
    eval_base_sora(prompt_list)
    print(f'Sora PAB246 50 steps')
    eval_pab_sora(prompt_list)
    
    with open('../../../assets/texts/prompt_ucf.txt', "r") as f:
        prompt_list = [line.strip() for line in f.readlines()]
    
    print(f'VBench Foresight 50 steps')
    #eval_base_ucf(prompt_list)
    print(f'UCF PAB246 50 steps')
    eval_pab_ucf(prompt_list)
    #eval_pab3(prompt_list)
    '''
