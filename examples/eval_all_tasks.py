import os
from easy_runner import EasyRunner


# ======================================
# 启动器：批量生成 eval 命令
# ======================================
if __name__ == "__main__":
    exp_name = "eval_all"
    runner = EasyRunner(log_name=exp_name)

    base_log_dir = "logs"  # 👈 替换为你的实际路径，如 /hdd/xqh/OSRL/logs

    task_list = [
        # bullet safety gym:
        # "OfflineAntCircle-v0",
        # "OfflineAntRun-v0",
        # "OfflineBallCircle-v0",
        # "OfflineBallRun-v0",
        # "OfflineCarRun-v0",
        # "OfflineCarCircle-v0",
        # "OfflineDroneCircle-v0",
        # "OfflineDroneRun-v0",
        
        # safety gymnasium:
        # "OfflineCarButton1Gymnasium-v0",
        # "OfflineCarButton2Gymnasium-v0",
        # "OfflineCarCircle1Gymnasium-v0",
        # "OfflineCarCircle2Gymnasium-v0",
        # "OfflineCarGoal1Gymnasium-v0",
        # "OfflineCarGoal2Gymnasium-v0",
        # "OfflineCarPush1Gymnasium-v0",
        # "OfflineCarPush2Gymnasium-v0",
        # "OfflinePointButton1Gymnasium-v0",
        # "OfflinePointButton2Gymnasium-v0",
        # "OfflinePointCircle1Gymnasium-v0",
        # "OfflinePointCircle2Gymnasium-v0",
        # "OfflinePointGoal1Gymnasium-v0",
        # "OfflinePointGoal2Gymnasium-v0",
        # "OfflinePointPush1Gymnasium-v0",
        # "OfflinePointPush2Gymnasium-v0",
        "OfflineAntVelocityGymnasium-v1",
        # "OfflineHalfCheetahVelocityGymnasium-v1",
        # "OfflineHopperVelocityGymnasium-v1",
        # "OfflineSwimmerVelocityGymnasium-v1",
        # "OfflineWalker2dVelocityGymnasium-v1",
    ]

    all_eval_commands = []
    all_model_dirs = []
    num_gpus = 8  # 根据你机器调整
    gpu_ids = list(range(num_gpus))

    for i, task in enumerate(task_list):
        task += "-cost-10"
        # task_dir = os.path.join(base_log_dir, "BallCircle-with-Q-C")
        task_dir = os.path.join(base_log_dir, task)
        if not os.path.isdir(task_dir):
            print(f"Task dir not found: {task_dir}, skipping...")
            continue

        # 列出该任务下所有子目录（每个是一个训练 run）
        model_dirs = [os.path.join(task_dir, d, d) for d in os.listdir(task_dir)
                      if os.path.isdir(os.path.join(task_dir, d, d))]

        for dir in model_dirs:
            all_model_dirs.append(dir)

    
    for i, dir in enumerate(all_model_dirs):
        gpu_id = gpu_ids[i % num_gpus]
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} nohup python examples/eval/eval_c2dt.py --path {dir} > {dir}/eval_result.txt 2>&1"
        all_eval_commands.append(cmd)

    print(f"Total eval jobs: {len(all_eval_commands)}")
    runner.start(all_eval_commands, max_parallel=24)
