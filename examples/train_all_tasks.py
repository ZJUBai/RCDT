from easy_runner import EasyRunner
import sys
if __name__ == "__main__":

    exp_name = "benchmark"
    runner = EasyRunner(log_name=exp_name)

    task = [
        # bullet safety gym envs
        # "OfflineAntCircle-v0",
        # "OfflineAntRun-v0",
        # "OfflineBallCircle-v0",
        # "OfflineBallRun-v0",
        # "OfflineCarRun-v0",
        # "OfflineCarCircle-v0",
        # "OfflineDroneCircle-v0",
        # "OfflineDroneRun-v0",

        # # safety gymnasium: car
        # "OfflineCarButton1Gymnasium-v0",
        # "OfflineCarButton2Gymnasium-v0",
        # "OfflineCarCircle1Gymnasium-v0",
        # "OfflineCarCircle2Gymnasium-v0",
        # "OfflineCarGoal1Gymnasium-v0",
        # "OfflineCarGoal2Gymnasium-v0",
        # "OfflineCarPush1Gymnasium-v0",
        # "OfflineCarPush2Gymnasium-v0",

        # # safety gymnasium: velocity
        # "OfflineAntVelocityGymnasium-v1",
        "OfflineHalfCheetahVelocityGymnasium-v1",
        "OfflineHopperVelocityGymnasium-v1",
        "OfflineSwimmerVelocityGymnasium-v1",
        "OfflineWalker2dVelocityGymnasium-v1",

        # # safety gymnasium: point
        "OfflinePointButton1Gymnasium-v0",
        "OfflinePointButton2Gymnasium-v0",
        "OfflinePointCircle1Gymnasium-v0",
        "OfflinePointCircle2Gymnasium-v0",
        "OfflinePointGoal1Gymnasium-v0",
        "OfflinePointGoal2Gymnasium-v0",
        "OfflinePointPush1Gymnasium-v0",
        "OfflinePointPush2Gymnasium-v0",

        # # metadrive envs
        # "OfflineMetadrive-easysparse-v0",
        # "OfflineMetadrive-easymean-v0",
        # "OfflineMetadrive-easydense-v0",
        # "OfflineMetadrive-mediumsparse-v0",
        # "OfflineMetadrive-mediummean-v0",
        # "OfflineMetadrive-mediumdense-v0",
        # "OfflineMetadrive-hardsparse-v0",
        # "OfflineMetadrive-hardmean-v0",
        # "OfflineMetadrive-harddense-v0",
    ]


    policy = ["train_c2dt"]

    critic_lr = [5e-5]
    tau       = [1e-2]
    eta_q     = [0.1,0.5,1]
    eta_c     = [0.1,0.5,1]
    discount  = [0.9]
    alpha     = [1,5]

    num_gpus = 8
    first_gpu_id = 0
    gpu_ids = list(range(num_gpus))  # [0, 1, 2, ..., 7]

    all_combinations = [(p, t, lr, ta, etaq, etac, dis, a)  for t in task for lr in critic_lr for ta in tau for etaq in eta_q for etac in eta_c for a in alpha for dis in discount for p in policy]
    # all_combinations = [(p,t,s) for t in task for p in policy for s in seed]
    train_commands = []
    for i, (p, t, lr, ta, etaq, etac, dis, a) in enumerate(all_combinations):
        gpu_id = gpu_ids[(i+first_gpu_id) % num_gpus]
        # 强制只看到指定 GPU，并让程序使用 cuda:0（因为可见设备被限制了）
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} nohup python examples/train/{p}.py --task {t} --eta_q {etaq} --eta_c {etac} --alpha {a} --discount {dis} --device cuda:0 --critic_lr {lr} --tau {ta}"
        # cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} nohup python examples/train/{p}.py --task {t} --device cuda:0 --seed {s}"
        train_commands.append(cmd)

    runner.start(train_commands, max_parallel=16)

