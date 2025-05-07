![这是关于python](
https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSu_Z0odJUp5ZwZMaMQoUo7N_EFhKkJDOie7Q&s
)
# 1. 创建环境
# 注意：目前环境仅在mac pro max实验，其他平台暂未测试

## 方法：通过创建新环境并迁移依赖
### 1. 激活原虚拟环境
    source .venv_test/bin/activate
### 2. 在虚拟环境激活的情况下，导出当前环境的依赖列表到一个 requirements.txt 文件
    pip freeze > requirements.txt
### 3. 退出当前虚拟环境
    deactivate
### 4. 创建新的虚拟环境
    python3 -m venv new_venv_name
### 5. 激活新的虚拟环境
    source new_venv_name/bin/activate
### 6. 使用 requirements.txt 文件来安装之前虚拟环境中的所有依赖
    pip install -r requirements.txt
### 7. 确保新的虚拟环境一切正常，可以再次使用
    python --version

# 2. 下载mujoco模型库
## 创建env_scenes
```bash
mkdir env_scenes
cd env_scenes
```
## 拉取mujoco模型库
```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git
```