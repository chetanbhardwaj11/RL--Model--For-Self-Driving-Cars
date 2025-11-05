import os
import threading
import random
import tkinter as tk
import customtkinter as ctk
from collections import defaultdict
import numpy as np
from PIL import Image, ImageTk, ImageOps
import pickle
import time

# Config
QTABLE_PATH = "qtable.pkl"   # file to save/load Q-table
CANVAS_BG = "#1f1f28"

# Helper: load png and composite onto canvas background to preserve transparency
def hex_to_rgb(hexcolor: str):
    hexcolor = hexcolor.lstrip('#')
    return tuple(int(hexcolor[i:i+2], 16) for i in (0, 2, 4))

def load_png_on_bg(path, target_size, canvas_bg_hex):
    img = Image.open(path).convert("RGBA")
    img = ImageOps.contain(img, target_size, method=Image.LANCZOS)
    bg_rgb = hex_to_rgb(canvas_bg_hex)
    bg = Image.new("RGBA", img.size, (bg_rgb[0], bg_rgb[1], bg_rgb[2], 255))
    composed = Image.alpha_composite(bg, img)
    composed_rgb = composed.convert("RGB")
    return ImageTk.PhotoImage(composed_rgb)

# Simple training simulator env (used for qlearn_train and discretization)
class SimpleRoadEnv:
    def __init__(self, road_length=200.0, lanes=3, max_speed=8.0, dt=0.5, seed=None):
        self.road_length = road_length
        self.lanes = lanes
        self.max_speed = max_speed
        self.dt = dt
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        self.pos = 0.0
        self.speed = 4.0
        self.lane = self.lanes // 2
        self.cars = []
        num = self.rng.randint(2, 5)
        spacing = max(30, (self.road_length - 40) / (num + 1))
        for i in range(1, num + 1):
            x = 20 + i * spacing + self.rng.uniform(-5, 5)
            lane = int(self.rng.choice(range(self.lanes)))
            spd = self.rng.uniform(2.0, 6.0)
            self.cars.append({'x': x, 'lane': lane, 'spd': spd})
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        dist = self.road_length
        for c in self.cars:
            if c['lane'] == self.lane and c['x'] > self.pos:
                dist = min(dist, c['x'] - self.pos)
        dist = min(dist, self.road_length)
        return np.array([self.pos, self.speed, self.lane, dist], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        accel = 0.0
        lane_change = 0
        if action == 1:
            accel = +1.5
        elif action == 2:
            accel = -3.0
        elif action == 3:
            lane_change = -1
        elif action == 4:
            lane_change = +1
        elif action == 5:
            accel = -10.0

        new_lane = self.lane + lane_change
        if 0 <= new_lane < self.lanes:
            self.lane = new_lane

        self.speed = max(0.0, min(self.max_speed, self.speed + accel * self.dt))
        prev_pos = self.pos
        self.pos += self.speed * self.dt

        for c in self.cars:
            c['x'] += c['spd'] * self.dt

        collision = False
        for c in self.cars:
            if c['lane'] == self.lane and abs(c['x'] - self.pos) < 2.0:
                collision = True
                break

        reward = (self.pos - prev_pos)
        reward -= 0.02
        reward -= 0.01 * abs(accel)

        done = False
        info = {}
        if collision:
            reward -= 150.0
            done = True
            info['reason'] = 'collision'
        if self.pos >= self.road_length:
            reward += 250.0
            done = True
            info['reason'] = 'goal'
        if self.steps >= 300:
            done = True
            info['reason'] = 'timeout'
        return self._get_obs(), float(reward), done, info

# TabularAgent: dynamic/growing Q-table with save/load
class TabularAgent:
    def __init__(self, bins, lanes=3, n_actions=6, seed=0):
        self.bins = bins
        self.lanes = lanes
        self.n_actions = n_actions
        self.Q = {}  # dict: state_tuple -> np.array(n_actions,)
        random.seed(seed)
        np.random.seed(seed)

    def _key(self, s_idx):
        return tuple(int(x) for x in s_idx)

    def get_q(self, s_idx):
        k = self._key(s_idx)
        if k not in self.Q:
            self.Q[k] = np.zeros(self.n_actions, dtype=np.float32)
        return self.Q[k]

    def discretize(self, obs, env):
        # obs: [pos, speed, lane, dist]
        pos, speed, lane, dist = obs
        pos_i = 0
        if self.bins.get('pos', 1) > 1:
            pos_i = min(int(pos / env.road_length * self.bins['pos']), self.bins['pos'] - 1)
        speed_i = 0
        if self.bins.get('speed', 1) > 1:
            speed_i = min(int(speed / env.max_speed * self.bins['speed']), self.bins['speed'] - 1)
        lane_i = int(lane)
        dist_i = min(int(dist / env.road_length * self.bins['dist']), max(0, self.bins['dist'] - 1))
        return (pos_i, speed_i, lane_i, dist_i)

    def act(self, s_idx, epsilon=0.0):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        q = self.get_q(s_idx)
        return int(np.argmax(q))

    def q_update(self, s_idx, a, r, s2_idx, alpha, gamma):
        q = self.get_q(s_idx)
        q_next = self.get_q(s2_idx)
        td = r + gamma * np.max(q_next) - q[a]
        q[a] += alpha * td

    def table_size(self):
        return len(self.Q)

    def save(self, filepath):
        try:
            with open(filepath, "wb") as f:
                pickle.dump(self.Q, f)
            return True
        except Exception as e:
            print("Failed to save Q-table:", e)
            return False

    def load(self, filepath):
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, "rb") as f:
                loaded = pickle.load(f)
            # basic validation: dict-like
            if isinstance(loaded, dict):
                self.Q = loaded
                return True
            return False
        except Exception as e:
            print("Failed to load Q-table:", e)
            return False

# qlearn_train: offline training (updates agent in-place)
def qlearn_train(agent, env, episodes=400, alpha=0.12, gamma=0.98,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.996, report_fn=None):
    rewards = []
    epsilon = eps_start
    for ep in range(1, episodes + 1):
        obs = env.reset()
        s_idx = agent.discretize(obs, env)
        done = False
        ep_r = 0.0
        while not done:
            a = agent.act(s_idx, epsilon=epsilon)
            obs2, r, done, info = env.step(a)
            s2_idx = agent.discretize(obs2, env)
            agent.q_update(s_idx, a, r, s2_idx, alpha, gamma)
            s_idx = s2_idx
            ep_r += r
        rewards.append(ep_r)
        epsilon = max(eps_end, epsilon * eps_decay)
        if report_fn is not None and (ep == 1 or ep % 25 == 0):
            avg20 = float(np.mean(rewards[-20:])) if len(rewards) >= 1 else float(ep_r)
            try:
                report_fn(ep, ep_r, avg20)
            except Exception:
                pass
    return rewards


# DrivingUI: full class with continuous online learning and save/load on exit
class DrivingUI:
    def __init__(self, master):
        self.master = master
        master.title("RL Self-Driving (persistent, online learning)")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # top controls
        self.frame = ctk.CTkFrame(master)
        self.frame.pack(side="top", fill="x", padx=8, pady=8)
        self.btn_train = ctk.CTkButton(self.frame, text="Start Training / Auto", command=self.start_training_thread)
        self.btn_reset = ctk.CTkButton(self.frame, text="Reset Game", command=self.reset_game)
        self.status_label = ctk.CTkLabel(self.frame, text="Status: Waiting | Q-states: 0")
        self.btn_train.grid(row=0, column=0, padx=6)
        self.btn_reset.grid(row=0, column=1, padx=6)
        self.status_label.grid(row=0, column=2, padx=8)

        # canvas
        self.canvas_bg = CANVAS_BG
        self.canvas_w = 520
        self.canvas_h = 700
        self.canvas = tk.Canvas(master, width=self.canvas_w, height=self.canvas_h, bg=self.canvas_bg, highlightthickness=0)
        self.canvas.pack(padx=8, pady=8)

        # lanes
        self.lanes = 3
        self.lane_x = [self.canvas_w * (i + 0.5) / self.lanes for i in range(self.lanes)]
        self.draw_track()

        # car fixed pos
        self.car_width = 56
        self.car_height = 96
        self.car_px = self.lane_x[self.lanes // 2]
        self.car_py = self.canvas_h - 140

        # image placeholders
        self.tk_car_img = None
        self.tk_obst_img = None
        self.car_is_image = False
        self.obst_is_image = False

        # locate images
        def find_existing_path(candidates):
            for p in candidates:
                if os.path.isabs(p):
                    if os.path.exists(p):
                        return p
                else:
                    cwd = os.getcwd()
                    bases = (cwd, os.path.join(cwd, "images"), "/mnt/data", ".")
                    for base in bases:
                        full = os.path.join(base, p) if not os.path.isabs(p) else p
                        if os.path.exists(full):
                            return full
            return None

        car_candidates = ["car.png", "Car.png", "/mnt/data/car.png"]
        obst_candidates = ["obstacle.png", "obstacles.png", "/mnt/data/obstacle.png", "/mnt/data/obstacle.png"]
        car_path = find_existing_path(car_candidates)
        obst_path = find_existing_path(obst_candidates)

        # load images (composited onto canvas bg)
        if car_path:
            try:
                self.tk_car_img = load_png_on_bg(car_path, (self.car_width, self.car_height), self.canvas_bg)
                self.car = self.canvas.create_image(self.car_px, self.car_py, image=self.tk_car_img, anchor='center')
                self.car_is_image = True
                print("[UI] Loaded car image:", car_path)
            except Exception as e:
                print("[UI] Car load failed:", e)
                self.car_is_image = False
        if not self.car_is_image:
            left = self.car_px - self.car_width//2
            right = self.car_px + self.car_width//2
            top = self.car_py - self.car_height//2
            bottom = self.car_py + self.car_height//2
            self.car = self.canvas.create_rectangle(left, top, right, bottom, fill="#00cc66")
            print("[UI] Using fallback rectangle for car.")

        obst_w = int(self.car_width * 0.9)
        obst_h = int(self.car_height * 0.45)
        if obst_path:
            try:
                self.tk_obst_img = load_png_on_bg(obst_path, (obst_w, obst_h), self.canvas_bg)
                self.obst_is_image = True
                print("[UI] Loaded obstacle image:", obst_path)
            except Exception as e:
                print("[UI] Obstacle load failed:", e)
                self.obst_is_image = False
        else:
            print("[UI] No obstacle image found; using rectangles.")

        # obstacles list
        self.obstacles = []
        self.spawn_timer = 0

        # agent & sim env
        self.bins = {'pos':1, 'speed':1, 'dist':8}
        self.agent = TabularAgent(self.bins, lanes=self.lanes, n_actions=6, seed=1)
        self.sim_env = SimpleRoadEnv(road_length=200.0, lanes=self.lanes, max_speed=8.0, dt=0.5, seed=123)

        # load Q-table if exists
        if os.path.exists(QTABLE_PATH):
            ok = self.agent.load(QTABLE_PATH)
            if ok:
                print("[UI] Loaded Q-table from", QTABLE_PATH)
            else:
                print("[UI] Failed to load Q-table; starting fresh.")
        else:
            print("[UI] No Q-table file found; starting fresh.")

        # modes & timing
        self.auto_mode = False
        self.training = False
        self.tick_dt = 30            # ms; lower -> smoother/faster
        self.obstacle_speed = 12     # pixels per tick
        # autosave interval
        self.last_autosave = time.time()
        self.autosave_interval = 10.0  # seconds

        # ensure save on close
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

        # start loop
        self.master.after(self.tick_dt, self.loop)

    # drawing helpers
    def draw_track(self):
        self.canvas.delete("track")
        w = self.canvas_w; h = self.canvas_h
        for i in range(self.lanes + 1):
            x = (i / self.lanes) * w
            self.canvas.create_line(x, 0, x, h, fill="#5a5a66", width=2, tags="track")
        for i in range(1, self.lanes):
            x = (i / self.lanes) * w
            for y in range(0, h, 20):
                self.canvas.create_line(x, y, x, y+10, fill="#888892", tags="track")

    def update_car_graphics(self):
        if self.car_is_image:
            self.canvas.coords(self.car, self.car_px, self.car_py)
        else:
            left = self.car_px - self.car_width//2
            right = self.car_px + self.car_width//2
            top = self.car_py - self.car_height//2
            bottom = self.car_py + self.car_height//2
            self.canvas.coords(self.car, left, top, right, bottom)

    def spawn_obstacle(self):
        lane = random.randrange(self.lanes)
        px = self.lane_x[lane]
        py = -80
        if self.obst_is_image and self.tk_obst_img is not None:
            iid = self.canvas.create_image(px, py, image=self.tk_obst_img, anchor='center')
            is_img = True
        else:
            w = int(self.car_width * 0.9); h = int(self.car_height * 0.45)
            iid = self.canvas.create_rectangle(px-w//2, py-h//2, px+w//2, py+h//2, fill="#ff4444")
            is_img = False
        self.obstacles.append({'px': px, 'py': py, 'lane': lane, 'id': iid, 'is_image': is_img})

    # background training thread
    def start_training_thread(self):
        if self.training:
            return
        self.training = True
        self.status_label.configure(text=f"Training... | Q-states: {self.agent.table_size()}")
        agent = self.agent; env = self.sim_env

        def report(ep, ep_r, avg20):
            self.master.after(1, lambda: self.status_label.configure(text=f"Training ep {ep} r {ep_r:.1f} avg20 {avg20:.1f} | Q-states: {agent.table_size()}"))

        def train_run():
            rewards = qlearn_train(agent, env, episodes=600, alpha=0.12, gamma=0.98,
                                   eps_start=1.0, eps_end=0.04, eps_decay=0.996, report_fn=report)
            self.training = False
            self.auto_mode = True
            self.master.after(1, lambda: self.status_label.configure(text=f"Trained → Auto | Q-states: {agent.table_size()}"))
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(6,3)); plt.plot(rewards); plt.title("Training rewards"); plt.grid(True); plt.show(block=False)
            except Exception:
                pass

        t = threading.Thread(target=train_run, daemon=True)
        t.start()

    def reset_game(self):
        for ob in list(self.obstacles):
            try:
                self.canvas.delete(ob['id'])
            except Exception:
                pass
        self.obstacles = []
        self.spawn_timer = 0
        self.car_px = self.lane_x[self.lanes // 2]
        self.update_car_graphics()
        self.sim_env.reset()
        self.status_label.configure(text=f"Reset | Q-states: {self.agent.table_size()}")

    # main loop: moves obstacles, does online learning while auto_mode
    def loop(self):
        # spawn obstacles
        self.spawn_timer += 1
        if self.spawn_timer > 12:
            self.spawn_timer = 0
            if len(self.obstacles) < 10 and random.random() < 0.9:
                self.spawn_obstacle()

        # move obstacles
        new_obs = []
        for ob in self.obstacles:
            ob['py'] += self.obstacle_speed
            if ob['is_image']:
                self.canvas.coords(ob['id'], ob['px'], ob['py'])
            else:
                w = int(self.car_width * 0.9); h = int(self.car_height * 0.45)
                self.canvas.coords(ob['id'], ob['px']-w//2, ob['py']-h//2, ob['px']+w//2, ob['py']+h//2)
            if ob['py'] < self.canvas_h + 120:
                new_obs.append(ob)
            else:
                try:
                    self.canvas.delete(ob['id'])
                except Exception:
                    pass
        self.obstacles = new_obs

        # AUTO mode: choose action and perform online Q-learning
        if self.auto_mode and not self.training:
            car_lane = self.closest_lane(self.car_px)
            # nearest obstacle ahead in same lane (pixel distance)
            dist_px = 9999
            for ob in self.obstacles:
                if ob['lane'] == car_lane and ob['py'] < self.car_py:
                    d = self.car_py - ob['py']
                    if d < dist_px:
                        dist_px = d
            # convert pixel distance to meters-like
            dist_m = (dist_px / (self.canvas_h)) * self.sim_env.road_length
            s_idx = self.agent.discretize((0.0, 0.0, car_lane, dist_m), self.sim_env)
            # small epsilon for exploration during online play
            a = self.agent.act(s_idx, epsilon=0.02)

            # apply action (only lane-changes affect UI)
            old_lane = car_lane
            if a == 3:
                new_lane = max(0, car_lane - 1); self.car_px = self.lane_x[new_lane]
            elif a == 4:
                new_lane = min(self.lanes - 1, car_lane + 1); self.car_px = self.lane_x[new_lane]
            else:
                new_lane = car_lane

            self.update_car_graphics()

            # compute new distance after action (approx)
            dist_px_new = 9999
            for ob in self.obstacles:
                if ob['lane'] == new_lane and ob['py'] < self.car_py:
                    d = self.car_py - ob['py']
                    if d < dist_px_new:
                        dist_px_new = d
            dist_m_new = (dist_px_new / (self.canvas_h)) * self.sim_env.road_length

            # reward shaping for online learning:
            # - big negative if very close (danger)
            # - small positive if distance increased (safer)
            # - small negative if distance decreased (risk)
            if dist_px_new < 40:
                r = -50.0
            else:
                # encourage increasing clearance
                r = 0.2 if dist_px_new > dist_px else -0.1

            s2_idx = self.agent.discretize((0.0, 0.0, new_lane, dist_m_new), self.sim_env)
            # update Q-table online
            self.agent.q_update(s_idx, a, r, s2_idx, alpha=0.12, gamma=0.98)

        # collision detection using canvas bbox
        car_bbox = self.canvas.bbox(self.car)
        collided = False
        for ob in self.obstacles:
            ob_bbox = self.canvas.bbox(ob['id'])
            if car_bbox and ob_bbox:
                if not (car_bbox[2] < ob_bbox[0] or car_bbox[0] > ob_bbox[2] or car_bbox[3] < ob_bbox[1] or car_bbox[1] > ob_bbox[3]):
                    collided = True
                    break

        if collided:
            try:
                self.canvas.delete("crash")
            except Exception:
                pass
            self.canvas.create_text(self.canvas_w//2, self.canvas_h//2, text="COLLISION", font=("Arial", 34), fill="yellow", tags="crash")
            # immediate reset (preserve Q-table)
            self.master.after(350, lambda: self.canvas.delete("crash"))
            self.reset_game()

        # update status label
        self.status_label.configure(text=f"Status: {'Auto' if self.auto_mode else 'Waiting'} | Q-states: {self.agent.table_size()}")

        # periodic autosave of Q-table (so abrupt termination loses less)
        now = time.time()
        if now - self.last_autosave > self.autosave_interval:
            try:
                self.agent.save(QTABLE_PATH)
            except Exception:
                pass
            self.last_autosave = now

        # schedule next tick
        self.master.after(self.tick_dt, self.loop)

    def closest_lane(self, px):
        d = [abs(px - lx) for lx in self.lane_x]
        return int(np.argmin(d))

    # save Q-table on exit
    def on_closing(self):
        try:
            ok = self.agent.save(QTABLE_PATH)
            if ok:
                print("[UI] Saved Q-table to", QTABLE_PATH)
            else:
                print("[UI] Failed to save Q-table.")
        except Exception as e:
            print("[UI] Exception saving Q-table:", e)
        try:
            self.master.destroy()
        except Exception:
            pass


# Run
if __name__ == "__main__":
    print("Starting RL Self-Driving UI (persistent Q-table + online learning).")
    print("Place car.png and obstacle.png next to this script or in /mnt/data to use them (optional).")
    root = ctk.CTk()
    ui = DrivingUI(root)
    root.geometry("560x820")
    root.mainloop()
