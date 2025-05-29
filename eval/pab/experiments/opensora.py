from utils import generate_func, read_prompt_list

from videosys import OpenSoraConfig, OpenSoraPABConfig, VideoSysEngine


def eval_base(prompt_list):
    config = OpenSoraConfig()
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./samples/opensora_base", loop=5)

def eval_pab_60(prompt_list):
    pab_config = OpenSoraPABConfig(spatial_range=2, temporal_range=4, cross_range=6)
    config = OpenSoraConfig(enable_pab=True, pab_config=pab_config, num_sampling_steps=60, enable_flash_attn=True)
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./samples/sora_opensora_pab246_60steps", loop=1)

def eval_pab_30(prompt_list):
    pab_config = OpenSoraPABConfig(spatial_range=2, temporal_range=4, cross_range=6)
    config = OpenSoraConfig(enable_pab=True, pab_config=pab_config, num_sampling_steps=30, enable_flash_attn=True)
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./samples/evalcrafter_opensora_720p_2s_pab246_30steps", loop=1)

def eval_pab1(prompt_list):
    config = OpenSoraConfig(enable_pab=True)
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./samples/opensora_pab1", loop=5)


def eval_pab2(prompt_list):
    pab_config = OpenSoraPABConfig(spatial_gap=3, temporal_gap=5, cross_gap=7)
    config = OpenSoraConfig(enable_pab=True, pab_config=pab_config)
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./samples/opensora_pab2", loop=5)


def eval_pab3(prompt_list):
    pab_config = OpenSoraPABConfig(spatial_gap=5, temporal_gap=7, cross_gap=9)
    config = OpenSoraConfig(enable_pab=True, pab_config=pab_config)
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./samples/opensora_pab3", loop=5)


if __name__ == "__main__":
    #prompt_list = read_prompt_list("vbench/VBench_full_info_50.json")
    with open('../../../assets/texts/prompt_evalcrafter.txt', "r") as f:
        prompt_list = [line.strip() for line in f.readlines()]
    #eval_base(prompt_list)
    #eval_pab1(prompt_list)
    #eval_pab2(prompt_list)
    #eval_pab3(prompt_list)
    eval_pab_30(prompt_list)
    #print(f'\n\n60 steps')
    #eval_pab_60(prompt_list)
