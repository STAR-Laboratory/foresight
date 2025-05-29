from videosys import CogVideoXConfig, CogVideoXFORESIGHTConfig, VideoSysEngine


def run_base():
    # models: "THUDM/CogVideoX-2b" or "THUDM/CogVideoX-5b"
    # change num_gpus for multi-gpu inference
    config = CogVideoXConfig("THUDM/CogVideoX-2b", num_gpus=1)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    # num frames should be <= 49. resolution is fixed to 720p.
    # seed=-1 means random seed. >0 means fixed seed.
    video = engine.generate(
        prompt=prompt,
        guidance_scale=6,
        num_inference_steps=50,
        num_frames=17,
        seed=1024,
    ).video[0]
    engine.save_video(video, f"./outputs/{prompt}_baseline.mp4")


def run_pab():
    config = CogVideoXConfig("THUDM/CogVideoX-2b", enable_pab=True)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(
        prompt=prompt,
        guidance_scale=6,
        num_inference_steps=50,
        num_frames=17,
        seed=1024,
    ).video[0]
    engine.save_video(video, f"./outputs/{prompt}_pab.mp4")


def run_foresight():
    foresight_config = CogVideoXFORESIGHTConfig(
        warmup=8,
        recalculate=2,
        threshold=0.5,
    )
    config = CogVideoXConfig(
        "THUDM/CogVideoX-2b", enable_foresight=True, foresight_config=foresight_config
    )
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(
        prompt=prompt,
        guidance_scale=6,
        num_inference_steps=50,
        num_frames=17,
        seed=1024,
    ).video[0]
    engine.save_video(video, f"./outputs/{prompt}_foresight.mp4")


def run_low_mem():
    config = CogVideoXConfig("THUDM/CogVideoX-2b", cpu_offload=True, vae_tiling=True)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")


if __name__ == "__main__":
    # run_base()
    # run_pab()
    # run_low_mem()
    run_foresight()
