import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from pynput import keyboard

class BlueROVControl(Node):
    def __init__(self):
        super().__init__('bluerov_control')
        # Create 4 publishers for the horizontal thrusters
        self.pubs = [self.create_publisher(Float64, f'/model/bluerov2/joint/thruster{i}_joint/cmd_thrust', 10) for i in range(1, 5)]
        
    def send_thrust(self, values):
        for i, val in enumerate(values):
            msg = Float64()
            msg.data = float(val)
            self.pubs[i].publish(msg)
            self.get_logger().info(f'Publishing to Thruster {i+1}: {val}')

    def on_press(self, key):
        try:
            if key.char == 'w':
                self.send_thrust([-10.0, -10.0, 10.0, 10.0])
            elif key.char == 's':
                self.send_thrust([10.0, 10.0, -10.0, -10.0])
        except AttributeError:
            pass

    def on_release(self, key):
        self.send_thrust([0.0, 0.0, 0.0, 0.0])
        if key == keyboard.Key.esc:
            return False

def main():
    rclpy.init()
    node = BlueROVControl()
    
    # Start the keyboard listener in a non-blocking way
    listener = keyboard.Listener(on_press=node.on_press, on_release=node.on_release)
    listener.start()

    print("Control Node Active. Press W/S to move. ESC to quit.")
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()