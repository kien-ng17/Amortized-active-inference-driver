# visualization.py — plotting helpers for all paper figures.
# Every function takes a ped_cross parameter (bool) to select the scenario.

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def _force_ped_cross(env, ped_cross):
    """Force env.ped_cross after a reset() call."""
    env.ped_cross = ped_cross


# training curves

def plot_training_curves(log_dir="sb3_logs", save_path="AIFmodetrain.png"):
    """Plot episode reward mean vs timesteps for the three training runs."""
    import pandas as pd
    df  = pd.read_csv(f"{log_dir}/AIF_stastic/progress.csv")
    df2 = pd.read_csv(f"{log_dir}/AIF_mode/progress.csv")
    df3 = pd.read_csv(f"{log_dir}/AIF/progress.csv")

    plt.plot(df["time/total_timesteps"],  df["rollout/ep_rew_mean"],
             label="Active Inference with static belief")
    plt.plot(df2["time/total_timesteps"], df2["rollout/ep_rew_mean"],
             label="Active Inference with the mode of belief")
    plt.plot(df3["time/total_timesteps"], df3["rollout/ep_rew_mean"],
             label="Full Active Inference")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward ")
    plt.title(" Acitve Inference (Mode of Belief) Training Progress")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.show()


# single-run velocity + belief dual-axis plot

def plot_velocity_belief(ego_model, env, save_path="Prior05.png", ped_cross=True):
    """Single-run dual-axis plot: car velocity and belief over one episode (Fig. velocity-belief)."""
    obs, inf = env.reset()
    _force_ped_cross(env, ped_cross)
    pos = []
    pos.append((env.ego_velocity, env.pf.particles[:, 0].mean()))
    step = 0
    done = False
    total_reward = 0
    while not done:
        action, _states = ego_model.predict(obs, deterministic=True)
        obs, reward, done, _, loc = env.step(action)
        pos.append((env.ego_velocity, env.pf.particles[:, 0].mean()))
        print(f"{obs}                  {env.reward}              {reward}         {env.step_env} ")
        step += 1
        total_reward += reward
    print(total_reward)
    pos = np.array(pos)
    print(pos[:, 0])

    fig, ax1 = plt.subplots()
    data1 = pos[:, 0]
    data2 = pos[:, 1]
    t = range(step + 1)
    color = 'tab:red'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('car velocity', color=color)
    ax1.plot(t, data1, 'x-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(10, 80)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('belief', color=color)
    ax2.plot(t, data2, 'o-', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title(f"prior belief {env.pf.fixed_prior} ( EFE= {total_reward})")
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    return total_reward


# False-belief single-run (prior != ground truth)

def plot_false_belief(ego_model, env, save_path="10False.png", ped_cross=True):
    """False-belief scenario: prior does not match ground truth (Fig. false-belief)."""
    obs, inf = env.reset()
    _force_ped_cross(env, ped_cross)
    pos = []
    pos.append((env.ego_velocity, env.pf.particles[:, 0].mean()))
    step = 0
    done = False
    total_reward = 0
    while not done:
        action, _states = ego_model.predict(obs, deterministic=False)
        obs, reward, done, _, loc = env.step(action)
        pos.append((env.ego_velocity, env.pf.particles[:, 0].mean()))
        print(f"{obs}                  {env.reward}              {reward}         {env.step_env} ")
        step += 1
        total_reward += reward
    print(total_reward)
    pos = np.array(pos)
    print(pos[:, 0])

    fig, ax1 = plt.subplots()
    data1 = pos[:, 0]
    data2 = pos[:, 1]
    t = range(step + 1)
    color = 'tab:red'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('car velocity', color=color)
    ax1.plot(t, data1, 'x-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('belief', color=color)
    ax2.plot(t, data2, 'o-', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title(
        f"Prior belief  ={env.pf.fixed_prior} , Ground truth = {int(ped_cross)} , "
        f"False Belief (EFE = {round(total_reward, 2)})"
    )
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    return total_reward


# prior-sweep cumulative-reward comparison (NoE vs AIF)

def plot_prior_sweep_reward(model_noe, model_aif, EnvCls, iter=500,
                             save_path="compareprior1.png", ped_cross=True):
    """Prior sweep: mean cumulative reward ± SEM for NoE vs AIF across iter episodes (Fig. reward-sweep)."""
    from scipy import stats as scipy_stats

    plt.figure(figsize=(12, 8))

    for ego_model, color_line, color_fill, label in [
        (model_noe, 'blue',  'lightblue', 'without Epistemic value'),
        (model_aif, 'red',   'lightpink', 'With Epistemic value'),
    ]:
        pos = []
        r = []
        for i in range(iter):
            env = EnvCls(ped_cross=ped_cross, prior_belief=i / iter)
            obs, inf = env.reset()
            _force_ped_cross(env, ped_cross)

            pos1 = []
            pos1.append(env.total_reward)
            step = 0
            done = False
            total_reward = 0
            while step < 9:
                action, _states = ego_model.predict(obs, deterministic=True)
                obs, reward, done, _, loc = env.step(action)
                pos1.append(env.total_reward)
                step += 1
                if not done:
                    total_reward += env.reward
            r.append(total_reward)
            pos.append(pos1)
            pos1 = np.array(pos1)
            t = range(step + 1)

        print(np.mean(r))
        mean_pos = np.mean(pos, axis=0)
        std_pos = scipy_stats.sem(pos, axis=0)
        print(mean_pos)
        print(std_pos)
        plt.plot(t, mean_pos, '--', color=color_line, label=label)
        plt.fill_between(t, mean_pos + std_pos, mean_pos - std_pos,
                         facecolor=color_fill, alpha=0.5)

    plt.xlabel('time step')
    plt.ylabel('Cumulatived reward ')
    plt.legend()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# velocity comparison: mode-of-belief vs AIF

def plot_prior_sweep_velocity(model_mode, model_aif, EnvCls, EnvCls2,
                               iter=100, save_path="compareprior1.png",
                               ped_cross=False):
    """Prior sweep: mean velocity ± std for mode-of-belief vs full-AIF policy (Fig. velocity-sweep)."""
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 22})

    for ego_model, EnvClass, prior, color_line, color_fill, label in [
        (model_mode, EnvCls2, 0.5,  'blue',  'lightblue', 'prior = 0.5'),
        (model_aif,  EnvCls,  0.5,  'red',   'lightpink', 'prior = 0.25'),
    ]:
        pos = []
        r = []
        for i in range(iter):
            env = EnvClass(ped_cross=ped_cross, prior_belief=prior)
            obs, inf = env.reset()
            _force_ped_cross(env, ped_cross)

            pos1 = []
            pos1.append(env.ego_velocity)
            step = 0
            done = False
            total_reward = 0
            while step < 10:
                action, _states = ego_model.predict(obs, deterministic=True)
                obs, reward, done, _, loc = env.step(action)
                pos1.append(env.ego_velocity)
                step += 1
                if not done:
                    total_reward += env.reward
            r.append(total_reward)
            pos.append(pos1)
            pos1 = np.array(pos1)
            t = range(step + 1)

        print(np.mean(r))
        mean_pos = np.mean(pos, axis=0)
        std_pos = np.std(pos, axis=0)
        print(mean_pos)
        plt.plot(t, mean_pos, '--', color=color_line, label=label)
        plt.fill_between(t, mean_pos + std_pos, mean_pos - std_pos,
                         facecolor=color_fill, alpha=0.5)

    plt.xlabel('time (s)')
    plt.ylabel('car velocity')
    plt.legend()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# trajectory snapshot subplots + belief/speed panel

def plot_ped(car_positions_x, car_positions_y, ped_positions_x,
             ped_positions_y, axs, i, ped_cross=True):
    """Draw one trajectory snapshot into axs[i]."""
    yielding = "pedestrian cross"
    if not ped_cross:
        yielding = "pedestrian don't cross"

    axs[i].plot(car_positions_x, car_positions_y,
                label='Car Trajectory', marker='o', color='blue')
    axs[i].plot(ped_positions_x, ped_positions_y,
                label='Pedestrian Trajectory', marker='x', color='red')
    axs[i].set_title(f'Timestep {len(car_positions_x) - 1}')
    axs[i].set_xlabel('X Position')
    axs[i].set_ylabel('Y Position')
    axs[i].set_ylim(-4, 10)
    axs[i].legend()
    axs[i].yaxis.tick_left()
    axs[i].grid(True)


def plot_trajectory_snapshots(ego_model, EnvCls, prior=0.5, ped_cross=True,
                               save_traj="IROS/traj.png",
                               save_belief="IROS/belief.png"):
    """Trajectory snapshots (3×1 subplots) + belief/speed panel (Fig. IROS trajectory)."""
    import os
    os.makedirs("IROS", exist_ok=True)

    plt.rcParams.update({'font.size': 20})

    env = EnvCls(ped_cross=ped_cross, prior_belief=prior)
    obs, _ = env.reset()
    _force_ped_cross(env, ped_cross)

    car_positions_x, car_positions_y = [], []
    ped_positions_x, ped_positions_y = [], []
    time_steps = []
    car_speed, ped_speed, belief = [], [], []

    print("Initial Observation:", obs)
    done = False
    t = 0
    fig, axs = plt.subplots(3, 1, layout='constrained', figsize=(8, 6))
    i = 0

    while not done:
        action, _ = ego_model.predict(obs, deterministic=True)
        obs, reward, done, _, loc = env.step(action)

        car_x = obs[0]
        car_y = 0
        car_speed.append(obs[1])
        ped_speed.append(env.ped_speed)
        ped_x = obs[2]
        ped_y = obs[3]
        b = obs[4]

        car_positions_x.append(car_x)
        car_positions_y.append(car_y)
        ped_positions_x.append(ped_x)
        ped_positions_y.append(ped_y)
        belief.append(b)

        if (t % 3 == 1) and i < 3:
            plot_ped(car_positions_x, car_positions_y,
                     ped_positions_x, ped_positions_y,
                     axs, i, env.ped_cross)
            i += 1

        time_steps.append(t)
        print(f"Time Step: {t}, State: {obs} - Action: {action} - "
              f"Conflict: {env.conflict} - Reward: {reward} - Done: {done}")
        t += 1

    # fill any remaining empty subplots with the final positions
    while i < 3:
        plot_ped(car_positions_x, car_positions_y,
                 ped_positions_x, ped_positions_y,
                 axs, i, env.ped_cross)
        i += 1

    plt.savefig(save_traj, bbox_inches='tight')
    plt.show()

    # ---- belief / speed panel ----
    fig, ax1 = plt.subplots(figsize=(9.5, 1.5))
    ax1.plot(belief, color='black')
    x = None
    idx = 1
    if idx < len(belief):
        x = ax1.plot(idx, belief[idx], "ro")
    while idx < len(belief):
        x = ax1.plot(idx, belief[idx], "ro")
        idx += 3
    if idx < 7 and idx < len(belief):
        idx += 2
        if idx < len(belief):
            x = ax1.plot(idx, belief[idx], "ro")
    ax1.set_ylabel('Belief')
    ax1.set_xlim(0, 8)
    ax1.set_ylim(-0.1, 1.1)

    ax2 = ax1.twinx()
    ax2.plot(car_speed, color='blue')
    idx = 1
    while idx < len(belief) and idx < len(car_speed):
        ax2.plot(idx, car_speed[idx], "ro")
        idx += 3
    ax2.set_ylabel('Speed', color="blue")
    ax2.set_xlabel('Time Steps')
    ax2.tick_params(axis='y', labelcolor="blue")
    ax2.set_ylim(20, 90)
    if x is not None:
        plt.legend((x,), ('time step 1,4,7',), loc='lower right')
    plt.savefig(save_belief, bbox_inches='tight')
    plt.show()


# epistemic-value velocity comparison (with vs without)

def plot_epistemic_comparison(model_aif, model_noe, EnvCls, prior=0.25,
                               iter=50, save_path="compareprior0.png",
                               ped_cross=False):
    """Velocity comparison: with vs without epistemic value over iter episodes (Fig. epistemic)."""
    plt.figure(figsize=(10, 8))

    for ego_model, color_line, color_fill, label in [
        (model_aif, 'blue', 'lightblue', 'with epistemic value'),
        (model_noe, 'red',  'lightpink', 'no epistemic value'),
    ]:
        pos = []
        length = []
        r = []
        for i in range(iter):
            env = EnvCls(ped_cross=ped_cross, prior_belief=prior)
            obs, inf = env.reset()
            _force_ped_cross(env, ped_cross)

            pos1 = []
            pos1.append(env.ego_velocity)
            step = 0
            done = False
            total_reward = 0
            while step < 8:
                action, _states = ego_model.predict(obs, deterministic=True)
                obs, reward, done, _, loc = env.step(action)
                pos1.append(env.ego_velocity)
                if done:
                    length.append(step)
                step += 1
                total_reward += reward

            pos.append(pos1)
            r.append(total_reward)
            pos1 = np.array(pos1)
            t = range(step + 1)

        print(env.ped_cross)
        print(np.mean(r))
        print(np.mean(length))
        mean_pos = np.mean(pos, axis=0)
        std_pos = np.std(pos, axis=0)
        print(mean_pos)
        plt.xlabel('time (s)')
        plt.ylabel('car velocity')
        plt.tick_params(axis='y')
        plt.plot(t, mean_pos, '--', color=color_line, label=label)
        plt.fill_between(t, mean_pos + std_pos, mean_pos - std_pos,
                         facecolor=color_fill, alpha=0.5)

    plt.legend()
    plt.title(f"Compare policy with epistemic value and without , Ground truth = {ped_cross}")
    plt.ylim(20, 90)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
