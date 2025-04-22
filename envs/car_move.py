import pybullet as p
import pybullet_data

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

# 保持仿真运行
while True:
    # 设置小车的线速度，让它沿 X 轴移动
    p.resetBaseVelocity(objectUniqueId=r2d2_id, linearVelocity=[1, 0, 0], angularVelocity=[0, 0, 0])

    # 执行仿真步进
    p.stepSimulation()

    # 动态更新摄像机位置
    try:
        pos, _ = p.getBasePositionAndOrientation(r2d2_id)
        p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=pos)
    except Exception as e:
        print("Error updating camera position:", e)
