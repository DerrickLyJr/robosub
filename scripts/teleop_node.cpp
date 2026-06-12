#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/float64.hpp>
#include <termios.h>
#include <unistd.h>
#include <vector>
#include <iostream>

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
        // Publish unified velocity intent instead of raw hardware lines
        cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
        
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(50),
            std::bind(&BlueROVTeleop::timer_callback, this));

        twist_cmd_.linear.x = 0.0;
        twist_cmd_.linear.y = 0.0;
        twist_cmd_.linear.z = 0.0;
        twist_cmd_.angular.z = 0.0;
        
        RCLCPP_INFO(this->get_logger(), "Teleop Channel Unified. Sending velocity intent targets via /cmd_vel.");
    }

    void timer_callback() {
        cmd_vel_pub_->publish(twist_cmd_);
    }

   void run() {
        std::cout << "\n=============================================\n";
        std::cout << "  W/S: Forward/Back | A/D: Strafe Left/Right\n";
        std::cout << "  U/I: Climb/Dive   | Space: All Stop\n";
        std::cout << "=============================================\n";

        while (rclcpp::ok()) {
            rclcpp::spin_some(this->get_node_base_interface());

            char key = getch();
            
            // FIXED: Keys now modify ONLY their designated axis channels,
            // preserving state across other running vector directions!
            if (key == 'w')      { twist_cmd_.linear.x = 20.0; }
            else if (key == 's') { twist_cmd_.linear.x = -20.0; }
            else if (key == 'd') { twist_cmd_.linear.y = 20.0; }
            else if (key == 'a') { twist_cmd_.linear.y = -20.0; }
            
            else if (key == 'u') { 
                twist_cmd_.linear.z = 0.5;
                cmd_vel_pub_->publish(twist_cmd_); // Publish the step input immediately
                twist_cmd_.linear.z = 0.0;         // INSTANTLY drop the latch back to zero!
            }   
            else if (key == 'i') { 
                twist_cmd_.linear.z = -0.5;
                cmd_vel_pub_->publish(twist_cmd_); // Publish the step input immediately
                twist_cmd_.linear.z = 0.0;         // INSTANTLY drop the latch back to zero!
            }
            
            // All Stop explicit reset handle
            else if (key == ' ') { 
                twist_cmd_.linear.x = 0.0;   
                twist_cmd_.linear.y = 0.0;   
                twist_cmd_.linear.z = 0.0;  
                twist_cmd_.angular.z = 0.0; 
            }
            else if (key == 27) break; 
        }
    }

private:
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    geometry_msgs::msg::Twist twist_cmd_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<BlueROVTeleop>();
    node->run();
    if (rclcpp::ok()) {
        rclcpp::shutdown();
    }
    return 0;
}