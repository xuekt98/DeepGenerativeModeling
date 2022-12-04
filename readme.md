## 以运行DDPM为例
修改configs对应的配置文件，运行指令
```
python3 main.py --config config/DDPM/DDPM.yml
```
如果要加载断点，修改config中
```
model_load_path: your/model/checkpoint/path
optim_sche_load_path: your/optimizer and scheduler/checkpoint/path
```
如果要运行其他模型，修改config中
```
runner: your_runner
```
