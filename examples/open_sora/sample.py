from videosys import (
    OpenSoraConfig,
    OpenSoraPABConfig,
    OpenSoraFORESIGHTConfig,
    VideoSysEngine,
)


def run_base():
    # change num_gpus for multi-gpu inference
    # sampling parameters are defined in the config
    config = OpenSoraConfig(num_sampling_steps=30, cfg_scale=7.0, num_gpus=1)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    # num frames: 2s, 4s, 8s, 16s
    # resolution: 144p, 240p, 360p, 480p, 720p
    # aspect ratio: 9:16, 16:9, 3:4, 4:3, 1:1
    # seed=-1 means random seed. >0 means fixed seed.
    video = engine.generate(
        prompt=prompt,
        resolution="240p",
        aspect_ratio="9:16",
        num_frames="2s",
        seed=1024,
    ).video[0]
    engine.save_video(video, f"./outputs/{prompt}_baseline.mp4")


def run_low_mem():
    config = OpenSoraConfig(cpu_offload=True, tiling_size=1)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")


def run_pab():
    pab_config = OpenSoraPABConfig(spatial_range=2, temporal_range=4, cross_range=6)
    config = OpenSoraConfig(
        enable_pab=True,
        pab_config=pab_config,
        num_sampling_steps=30,
        enable_flash_attn=True,
    )
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(
        prompt=prompt,
        resolution="240p",
        aspect_ratio="9:16",
        num_frames="2s",
        seed=1024,
    ).video[0]
    engine.save_video(video, f"./outputs/{prompt}_pab.mp4")


def run_foresight():
    foresight_config = OpenSoraFORESIGHTConfig(
        warmup=8,
        recalculate=2,
        threshold=1,
    )
    config = OpenSoraConfig(
        enable_foresight=True,
        foresight_config=foresight_config,
        num_sampling_steps=30,
        enable_flash_attn=True,
    )
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(
        prompt=prompt,
        guidance_scale=7.5,
        num_inference_steps=50,
        seed=1024,
    ).video[0]
    engine.save_video(video, f"./outputs/{prompt}_foresight.mp4")


if __name__ == "__main__":
    # run_base()
    # run_low_mem()
    # run_pab()
    run_foresight()
