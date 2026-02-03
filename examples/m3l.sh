FOUNDRY_CHECKPOINTS_DIR=./ckpts \
  PYTHONNOUSERSITE=1 PYTHONPATH= \
  pixi run -e dev -- rfd3 design \
    out_dir=logs/inference_outs/demo/m3l_burried_long \
    inputs=examples/m3l.json \
    ckpt_path=rfd3 \
    inference_engine=rfdiffusion3 \
    dump_trajectories=True