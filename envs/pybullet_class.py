import pybullet as p
import pybullet_data
import math
import time
import random


# 基础仿真类
class BaseSimulation:
    def __init__(self, time_step=1 / 240, gravity=-9.8):
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, gravity)
        p.setTimeStep(time_step)
        self.objects = []

    def load_model(self, urdf_path, start_pos, start_orientation=[0, 0, 0]):
        """加载模型"""
        model_id = p.loadURDF(urdf_path, start_pos, p.getQuaternionFromEuler(start_orientation))
        self.objects.append(model_id)
        return model_id

    def set_camera(self, target_id, distance=3, yaw=50, pitch=-35):
        """动态调整摄像机视角"""
        pos, _ = p.getBasePositionAndOrientation(target_id)
        p.resetDebugVisualizerCamera(cameraDistance=distance, cameraYaw=yaw, cameraPitch=pitch,
                                     cameraTargetPosition=pos)

    def step(self):
        """执行仿真步进"""
        p.stepSimulation()

    def close(self):
        """断开连接"""
        p.disconnect()


# 圆形轨迹仿真
class CircularMotionSimulation(BaseSimulation):
    def __init__(self, radius=2.0, speed=5.0, center=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if center is None:
            center = [0, 0]
        self.radius = radius
        self.speed = speed
        self.center = center
        self.theta = 0.0

    def update_position(self, object_id):
        """更新小车的位置和朝向"""
        target_x = self.center[0] + self.radius * math.cos(self.theta)
        target_y = self.center[1] + self.radius * math.sin(self.theta)
        vx = -self.speed * math.sin(self.theta)
        vy = self.speed * math.cos(self.theta)
        yaw = math.atan2(vy, vx)
        p.resetBasePositionAndOrientation(
            bodyUniqueId=object_id,
            posObj=[target_x, target_y, 0.1],
            ornObj=p.getQuaternionFromEuler([0, 0, yaw])
        )
        self.theta += self.speed / self.radius * (1 / 240)


# 障碍物避让仿真
class ObstacleAvoidanceSimulation(BaseSimulation):
    def __init__(self, arena_size=5.0, speed=2.0, obstacle_count=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arena_size = arena_size
        self.speed = speed
        self.obstacles = []
        self.init_obstacles(obstacle_count)

    def init_obstacles(self, count):
        """随机生成障碍物"""
        for _ in range(count):
            x = random.uniform(-self.arena_size, self.arena_size)
            y = random.uniform(-self.arena_size, self.arena_size)
            obstacle_id = self.load_model("cube_small.urdf", [x, y, 0.1])
            self.obstacles.append(obstacle_id)

    def update_position(self, object_id):
        """更新小车的位置并避开障碍物"""
        pos, _ = p.getBasePositionAndOrientation(object_id)
        x, y, _ = pos

        # 随机移动方向
        dx = random.uniform(-1, 1) * self.speed * (1 / 240)
        dy = random.uniform(-1, 1) * self.speed * (1 / 240)

        # 检查是否与障碍物碰撞
        for obstacle_id in self.obstacles:
            obstacle_pos, _ = p.getBasePositionAndOrientation(obstacle_id)
            ox, oy, _ = obstacle_pos
            distance = math.sqrt((x + dx - ox) ** 2 + (y + dy - oy) ** 2)
            if distance < 0.5:  # 如果距离小于阈值，改变运动方向
                dx = -dx
                dy = -dy

        # 更新位置
        new_x = x + dx
        new_y = y + dy
        yaw = math.atan2(dy, dx)
        p.resetBasePositionAndOrientation(
            bodyUniqueId=object_id,
            posObj=[new_x, new_y, 0.1],
            ornObj=p.getQuaternionFromEuler([0, 0, yaw])
        )


# 主程序
if __name__ == "__main__":
    # 选择仿真模式
    mode = input("Choose simulation mode (1: Circular Motion, 2: Obstacle Avoidance): ")

    if mode == "1":
        # 圆形轨迹仿真
        sim = CircularMotionSimulation()
        plane_id = sim.load_model("plane.urdf", [0, 0, 0])
        r2d2_id = sim.load_model("r2d2.urdf", [0, 0, 0.1])

        while True:
            sim.update_position(r2d2_id)
            sim.set_camera(r2d2_id)
            sim.step()
            time.sleep(1 / 240)

    elif mode == "2":
        # 障碍物避让仿真
        sim = ObstacleAvoidanceSimulation()
        plane_id = sim.load_model("plane.urdf", [0, 0, 0])
        r2d2_id = sim.load_model("r2d2.urdf", [0, 0, 0.1])

        while True:
            sim.update_position(r2d2_id)
            sim.set_camera(r2d2_id)
            sim.step()
            time.sleep(1 / 240)

    else:
        print("Invalid mode selected.")
