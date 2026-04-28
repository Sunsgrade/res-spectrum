# 反应谱计算 API 说明

## 接口地址

POST /reaction_api/response-spectrum

## 请求参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---:|---|
| file | file | 是 | - | 地震波文件，支持 .txt / .csv |
| damp | float | 否 | 0.05 | 阻尼比 |
| dt | float | 是 | - | 地震波时间步长 |
| Tp | float | 否 | 6.0 | 最大周期 |
| dtp | float | 否 | 0.02 | 反应谱周期间隔 |

## 返回字段

| 字段 | 说明 |
|---|---|
| success | 是否计算成功 |
| task_id | 本次计算任务编号 |
| filename | 上传文件名 |
| params | 本次计算参数 |
| result.T | 周期 |
| result.ARS | 加速度反应谱 |
| result.VRS | 速度反应谱 |
| result.DRS | 位移反应谱 |
| result.PVRS | 伪速度谱 |
| result.PARS | 伪加速度谱 |
| download_url | CSV 下载地址 |