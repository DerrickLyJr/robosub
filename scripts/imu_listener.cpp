#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

class IMUListener : public rclcpp::Node {
public:
    IMUListener() : Node("imu_listener") {
        // FIXED: Subscribing to the explicit long track passing across your parameter bridge
        subscription_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/world/bluerov2_underwater/model/bluerov2/link/base_link/sensor/imu_sensor/imu", 10,
            std::bind(&IMUListener::imu_callback, this, std::placeholders::_1));
        
        RCLCPP_INFO(this->get_logger(), "IMU Listener Started. Awaiting data on long path...");
    }

private:
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) const {
        double ax = msg->linear_acceleration.x;
        double ay = msg->linear_acceleration.y;
        double az = msg->linear_acceleration.z;

        double gx = msg->angular_velocity.x;
        double gy = msg->angular_velocity.y;
        double gz = msg->angular_velocity.z;

        double orientation_variance_x = msg->orientation_covariance[0];
        double gyro_variance_z = msg->angular_velocity_covariance[8];      
        double accel_variance_x = msg->linear_acceleration_covariance[0];  

        RCLCPP_INFO(this->get_logger(), 
            "\n--- SENSOR MATRIX CHECK ---\n"
            "Accel X: %.4f (Variance: %.6f)\n"
            "Accel Y: %.4f\n"
            "Accel Z: %.4f\n"
            "Gyro Z:  %.4f (Variance: %.6f)\n"
            "Orientation Matrix [0]: %.6f",
            ax, accel_variance_x, ay, az, gz, gyro_variance_z, orientation_variance_x);
    }
    
    // FIXED: Moved the variable out of the callback function scope and into the class private variables scope
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subscription_;
};

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<IMUListener>());
    rclcpp::shutdown();
    return 0;
}