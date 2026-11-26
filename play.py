"""Play the seeded runner with a keyboard."""

import argparse
import tkinter as tk

from env import Action, DinoEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="course seed (default: 42)")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")

    env = DinoEnv(render_mode="rgb_array")
    env.reset(seed=args.seed)
    window = tk.Tk()
    window.title("Dino Runner")
    window.resizable(False, False)
    screen = tk.Label(window, borderwidth=0)
    screen.pack()
    controls = tk.Label(window, text="Space / ↑ Jump    ↓ Duck    R Restart    Esc Quit")
    controls.pack(padx=16, pady=(12, 2))
    status = tk.Label(window, text="Press Space to start")
    status.pack(padx=16, pady=(0, 12))

    keys = set()
    running = False
    done = False

    def draw():
        frame = env.render()
        height, width = frame.shape[:2]
        ppm = f"P6\n{width} {height}\n255\n".encode("ascii") + frame.tobytes()
        image = tk.PhotoImage(data=ppm, format="PPM").zoom(2)
        screen.configure(image=image)
        screen.image = image

    def key_down(event):
        nonlocal running, done
        if event.keysym == "Escape":
            window.destroy()
        elif event.keysym.lower() == "r" and event.keysym not in keys:
            keys.add(event.keysym)
            env.reset(seed=args.seed)
            running, done = False, False
            status.configure(text="Press Space to start")
            draw()
        elif event.keysym in ("space", "Up") and not running and not done:
            running = True
        else:
            keys.add(event.keysym)

    def tick():
        nonlocal running, done
        if running:
            action = Action.RUN
            if "Down" in keys:
                action = Action.DUCK
            elif "space" in keys or "Up" in keys:
                action = Action.JUMP

            _, _, crashed, finished, info = env.step(action)
            status.configure(text=f"Survived: {info['survival_ticks']} ticks")
            if crashed or finished:
                running, done = False, True
                outcome = f"Hit a {info['crash_type']}" if crashed else "Course complete"
                status.configure(text=f"{outcome} · {info['survival_ticks']} ticks · R to restart")
            draw()
        window.after(33, tick)

    window.bind("<KeyPress>", key_down)
    window.bind("<KeyRelease>", lambda event: keys.discard(event.keysym))
    window.bind("<FocusOut>", lambda _event: keys.clear())
    draw()
    tick()
    try:
        window.mainloop()
    finally:
        env.close()


if __name__ == "__main__":
    main()
