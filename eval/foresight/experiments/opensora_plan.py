from utils import generate_func, read_prompt_list, generate_func_opensora_plan

from videosys import OpenSoraPlanConfig, OpenSoraPlanV110PABConfig, VideoSysEngine


def eval_base(prompt_list):
    config = OpenSoraPlanConfig(version="v110", transformer_type="65x512x512", num_gpus=1)
    engine = VideoSysEngine(config)
    generate_func_opensora_plan(engine, prompt_list, "./samples/vbench_opensoraplan_baseline_65-512-512", loop=1)

def eval_base_v120(prompt_list):
    config = OpenSoraPlanConfig(version="v120", transformer_type="29x480p", num_gpus=1)
    engine = VideoSysEngine(config)
    generate_func_opensora_plan(engine, prompt_list, "./samples/vbench_opensoraplan_v120_baseline_29x480p", loop=1)


def eval_pab1(prompt_list):
    pab_config = OpenSoraPlanV110PABConfig(
        spatial_gap=2,
        temporal_gap=4,
        cross_gap=6,
    )
    config = OpenSoraPlanConfig(version="v110", transformer_type="65x512x512", enable_pab=True, pab_config=pab_config)
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./samples/opensoraplan_pab1", loop=5)


def eval_pab2(prompt_list):
    pab_config = OpenSoraPlanV110PABConfig(
        spatial_gap=3,
        temporal_gap=5,
        cross_gap=7,
    )
    config = OpenSoraPlanConfig(version="v110", transformer_type="65x512x512", enable_pab=True, pab_config=pab_config)
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./samples/opensoraplan_pab2", loop=5)


def eval_pab3(prompt_list):
    pab_config = OpenSoraPlanV110PABConfig(
        spatial_gap=5,
        temporal_gap=7,
        cross_gap=9,
    )
    config = OpenSoraPlanConfig(version="v110", transformer_type="65x512x512", enable_pab=True, pab_config=pab_config)
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./samples/opensoraplan_pab3", loop=5)


if __name__ == "__main__":
    prompt_list = read_prompt_list("vbench/VBench_full_info_50.json")
    eval_base_v120(prompt_list)
    #eval_pab1(prompt_list)
    #eval_pab2(prompt_list)
    #eval_pab3(prompt_list)
