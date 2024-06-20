
from minigrid.manual_control import ManualControl
from core.actions import Actions


class NewManualControl(ManualControl):
    def __init__(self, env, seed = None):
        super().__init__(env, seed)

    def step(self, action: str):
        _, reward, terminated, truncated, _ = self.env.step(action)
        print(f"step={self.env.step_count}, reward={reward:.2f}")

        if terminated:
            print("Terminated")
            self.reset()
        elif truncated:
            print("Truncated")
            self.reset(self.seed)
        else:
            self.env.render()

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.env.close()
            return
        if key == "backspace":
            self.reset()
            return
        
        key_to_action = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.up,
            "down": Actions.down
        }

        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        else:
            print(key)