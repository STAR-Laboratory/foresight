from utils import generate_func, read_prompt_list

from videosys import OpenSoraConfig, OpenSoraPABConfig, OpenSoraFORESIGHTConfig, VideoSysEngine

# ========== Baseline ============

def eval_base_evalcrafter(prompt_list):
    config = OpenSoraConfig()
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./evalcrafter_opensora_baseline_30steps", loop=1)

def eval_base_ucf(prompt_list):
    config = OpenSoraConfig()
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./ucf_opensora_baseline_30steps", loop=1)

def eval_base_sora(prompt_list):
    config = OpenSoraConfig()
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./sora_opensora_baseline_30steps", loop=1)

# ========== PAB ============

def eval_pab_evalcrafter(prompt_list):
    pab_config = OpenSoraPABConfig(
        spatial_range=2,
        temporal_range=4,
        cross_range=6,
    )
    config = OpenSoraConfig(enable_pab=True, pab_config=pab_config)
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./evalcrafter_opensora_pab246_30steps", loop=1)


def eval_pab_ucf(prompt_list):
    pab_config = OpenSoraPABConfig(
        spatial_range=2,
        temporal_range=4,
        cross_range=6,
    )
    config = OpenSoraConfig(enable_pab=True, pab_config=pab_config)
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./ucf_opensora_pab246_30steps", loop=1)

def eval_pab_sora(prompt_list):
    pab_config = OpenSoraPABConfig(
        spatial_range=2,
        temporal_range=4,
        cross_range=6,
    )
    config = OpenSoraConfig(enable_pab=True, pab_config=pab_config)
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./sora_opensora_pab246_30steps", loop=1)

# ========== Foresight ============

def eval_foresight_evalcrafter(prompt_list):
    foresight_config = OpenSoraFORESIGHTConfig(
        warmup=8,
        recalculate=2,
        threshold=0.5,
    )
    config = OpenSoraConfig(enable_foresight=True, foresight_config=foresight_config)
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./evalcrafter_opensora_foresight_8_2_0.5_30steps", loop=1)


def eval_foresight_ucf(prompt_list):
    foresight_config = OpenSoraFORESIGHTConfig(
        warmup=8,
        recalculate=2,
        threshold=1,
    )
    config = OpenSoraConfig(enable_foresight=True, foresight_config=foresight_config)
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./ucf_opensora_foresight_8_2_0.5_30steps", loop=1)

def eval_foresight_sora(prompt_list):
    foresight_config = OpenSoraFORESIGHTConfig(
        warmup=8,
        recalculate=2,
        threshold=1,
    )
    config = OpenSoraConfig(enable_foresight=True, foresight_config=foresight_config)
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./sora_opensora_foresight_8_2_0.5_30steps", loop=1)


def eval_foresight_vbench(prompt_list, warmup, recalculate, threshold):
    foresight_config = OpenSoraFORESIGHTConfig(
        warmup=warmup,
        recalculate=recalculate,
        threshold=threshold,
    )
    config = OpenSoraConfig(enable_foresight=True, foresight_config=foresight_config)
    engine = VideoSysEngine(config)
    output_dir = f'./vbench_opensora_foresight_{warmup}_{recalculate}_{threshold}_30steps'
    generate_func(engine, prompt_list, output_dir, loop=1)


if __name__ == "__main__":
    prompt_list = read_prompt_list("vbench/VBench_full_info_30.json")
    
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
                print(f'VBench Foresight Wamrup {w}, Recalculate {r}, Threshold {t} with 30 steps')
                eval_foresight_vbench(prompt_list, w, r, t)