#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>
#include <termios.h>
#include <unistd.h>
#include <vector>
#include <iomanip> // For std::setprecision

char getch() {
    char buf = 0;
    struct termios old = {0};
    if (tcgetattr(0, &old) < 0) perror("tcsetattr()");
    old.c_lflag &= ~ICANON;
    old.c_lflag &= ~ECHO;
    old.c_cc[VMIN] = 1;
    old.c_cc[VTIME] = 0;
    if (tcsetattr(0, TCSANOW, &old) < 0) perror("tcsetattr ICANON");
    if (read(0, &buf, 1) < 0) perror("read()");
    old.c_lflag |= ICANON;
    old.c_lflag |= ECHO;
    if (tcsetattr(0, TCSADRAIN, &old) < 0) perror("tcsetattr ~ICANON");
    return buf;
}

class BlueROVTeleop : public rclcpp::Node {
public:
    BlueROVTeleop() : Node("bluerov_teleop") {
        // 1. Set up SensorData QoS (Best Effort) to prevent message lag
        auto qos = rclcpp::SensorDataQoS();

        for (int i = 1; i <= 6; ++i) {
            std::string topic = "/model/bluerov2/joint/thruster" + std::to_string(i) + "_joint/cmd_thrust";
            publishers_.push_back(this->create_publisher<std_msgs::msg::Float64>(topic, 10));
        }

        // 2. Apply the same QoS to the subscriber
        depth_sub_ = this->create_subscription<std_msgs::msg::Float64>(
            "/model/bluerov2/calculated_depth", 10,
            std::bind(&BlueROVTeleop::depth_callback, this, std::placeholders::_1));

        // 3. Initialize the timer to publish commands at 20Hz
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(50),
            std::bind(&BlueROVTeleop::timer_callback, this));

        // Default command is STOP
        current_command_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        
        RCLCPP_INFO(this->get_logger(), "Teleop Optimized. Control loop at 20Hz.");
    }

    // This callback runs every 50ms regardless of keyboard input
    void timer_callback() {
        auto msg = std_msgs::msg::Float64();
        for (size_t i = 0; i < current_command_.size(); ++i) {
            msg.data = current_command_[i];
            publishers_[i]->publish(msg);
        }
    }

    void depth_callback(const std_msgs::msg::Float64::SharedPtr msg) {
        this->current_depth_ = msg->data;
        std::cout << "\rCurrent Depth: " << std::fixed << std::setprecision(2) 
                  << current_depth_ << " m    " << std::flush;
    }

    void run() {
        while (rclcpp::ok()) {
            // Processing depth callbacks while waiting for keys
            rclcpp::spin_some(this->get_node_base_interface());

            char key = getch();
            if (key == 'w')      current_command_ = {-40.0, -40.0, 40.0, 40.0, 0.0, 0.0};
            else if (key == 's') current_command_ = {40.0, 40.0, -40.0, -40.0, 0.0, 0.0};
            else if (key == 'd') current_command_ = {40.0, -40.0, 40.0, -40.0, 0.0, 0.0};
            else if (key == 'a') current_command_ = {-40.0, 40.0, -40.0, 40.0, 0.0, 0.0};
            else if (key == 'u') current_command_ = {0.0, 0.0, 0.0, 0.0, -40.0, -40.0};
            else if (key == 'i') current_command_ = {0.0, 0.0, 0.0, 0.0, 40.0, 40.0};
            else if (key == ' ') current_command_ = {0, 0, 0, 0, 0, 0};
            else if (key == 27) break; 
        }
    }

private:
    std::vector<rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr> publishers_;
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr depth_sub_;
    rclcpp::TimerBase::SharedPtr timer_; // Timer handle
    std::vector<double> current_command_; // Stored state
    double current_depth_ = 0.0;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<BlueROVTeleop>();
    node->run();
    rclcpp::shutdown();
    return 0;
}