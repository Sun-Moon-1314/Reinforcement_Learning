import pybullet as p
import pybullet_data
import math
import time

# 连接到 PyBullet GUI
client = p.connect(p.GUI)

# 设置路径
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 初始化仿真环境
p.setGravity(0, 0, -9.8)  # 设置重力
p.setTimeStep(1/240)      # 设置时间步长
p.loadURDF("plane.urdf")  # 加载地面

# 加载小车模型
r2d2_id = p.loadURDF("r2d2.urdf", [0, 0, 0.1])  # 加载小车，初始高度为 0.1
if r2d2_id == -1:
    raise ValueError("Failed to load r2d2.urdf. Please check the file path or model definition.")

# 圆形轨迹参数
radius = 2.0               # 圆的半径
center = [0, 0]            # 圆心坐标
speed = 5.0                # 小车的线速度（单位：米/秒）

# 初始化角度
theta = 0.0                # 小车在圆上的初始角度（弧度制）

# 保持仿真运行
while True:
    # 计算小车在圆上的目标位置
    target_x = center[0] + radius * math.cos(theta)
    target_y = center[1] + radius * math.sin(theta)

    # 计算小车的速度方向（切线方向）
    vx = -speed * math.sin(theta)  # 速度在 X 方向的分量
    vy = speed * math.cos(theta)   # 速度在 Y 方向的分量

    # 计算小车的朝向角（yaw），使车头朝向速度方向
    yaw = math.atan2(vy, vx)  # 计算朝向角（弧度制）

    # 设置小车的线速度，不设置角速度（小车不自转）
    p.resetBasePositionAndOrientation(
        bodyUniqueId=r2d2_id,  # 小车的唯一 ID
        posObj=[target_x, target_y, 0.1],  # 小车的当前位置
        ornObj=p.getQuaternionFromEuler([0, 0, yaw])  # 将 yaw 转换为四元数
    )
    # p.resetBaseVelocity(
    #     objectUniqueId=r2d2_id,
    #     linearVelocity=[vx, vy, 0],  # 线速度
    #     angularVelocity=[0, 0, yaw]   # 不自转
    # )

    # 更新角度，模拟小车沿圆形轨迹前进
    theta += speed / radius * (1/240)  # 根据弧长公式更新角度（Δθ = v / r * Δt）

    # 执行仿真步进
    p.stepSimulation()

    # 动态更新摄像机位置
    try:
        pos, _ = p.getBasePositionAndOrientation(r2d2_id)
        p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=pos)
    except Exception as e:
        print("Error updating camera position:", e)

    # 让仿真更平滑
    time.sleep(1/240)
