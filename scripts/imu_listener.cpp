#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

class IMUListener : public rclcpp::Node {
public:
    IMUListener() : Node("imu_listener") {
        // Subscribe to the IMU topic
        subscription_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/model/bluerov2/sensor/imu_sensor/imu", 10,
            std::bind(&IMUListener::imu_callback, this, std::placeholders::_1));
        
        RCLCPP_INFO(this->get_logger(), "IMU Listener Started. Awaiting data...");
    }

private:
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) const {
        // Linear Acceleration (m/s^2)
        double ax = msg->linear_acceleration.x;
        double ay = msg->linear_acceleration.y;
        double az = msg->linear_acceleration.z;

        // Angular Velocity (rad/s)
        double gx = msg->angular_velocity.x;
        double gy = msg->angular_velocity.y;
        double gz = msg->angular_velocity.z;

        RCLCPP_INFO(this->get_logger(), 
            "\n--- IMU DATA ---\nAccel: [%.2f, %.2f, %.2f]\nGyro:  [%.2f, %.2f, %.2f]",
            ax, ay, az, gx, gy, gz);
    }
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subscription_;
};

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<IMUListener>());
    rclcpp::shutdown();
    return 0;
}