# api.py
import os
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.reaction_core import (
    load_wave_file,
    calculate_response_spectrum,
    save_result_csv,
)

app = FastAPI(
    title="Response Spectrum API",
    description="反应谱在线计算接口",
    version="0.1.0",
)

# 允许前端跨域访问。开发阶段可以先放开，正式部署时再改成公司官网域名。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)


@app.get("/")
def index():
    return {
        "message": "Response Spectrum API is running.",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }


@app.post("/api/response-spectrum")
async def response_spectrum(
    file: UploadFile = File(...),
    damp: float = Form(0.05),
    dt: float = Form(...),
    Tp: float = Form(6.0),
    dtp: float = Form(0.02),
):
    """
    上传地震波文件并计算反应谱。

    参数：
    - file: txt/csv 地震波文件
    - damp: 阻尼比
    - dt: 时间步长
    - Tp: 最大周期
    - dtp: 周期间隔
    """

    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in [".txt", ".csv"]:
        raise HTTPException(
            status_code=400,
            detail="只支持 .txt 或 .csv 文件。",
        )

    task_id = str(uuid.uuid4())
    input_path = os.path.join(TEMP_DIR, f"{task_id}{file_ext}")
    output_path = os.path.join(TEMP_DIR, f"{task_id}_result.csv")

    try:
        # 保存上传文件
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)

        # 读取地震波数据
        wave = load_wave_file(input_path)

        # 调用核心算法
        result = calculate_response_spectrum(
            wave=wave,
            damp=damp,
            dt=dt,
            Tp=Tp,
            dtp=dtp,
        )

        # 保存 CSV，方便前端下载
        save_result_csv(result, output_path)

        return {
            "success": True,
            "task_id": task_id,
            "filename": file.filename,
            "params": {
                "damp": damp,
                "dt": dt,
                "Tp": Tp,
                "dtp": dtp,
            },
            "result": result,
            "download_url": f"/api/response-spectrum/download/{task_id}",
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"计算失败：{str(e)}")


@app.get("/api/response-spectrum/download/{task_id}")
def download_result(task_id: str):
    output_path = os.path.join(TEMP_DIR, f"{task_id}_result.csv")

    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="结果文件不存在。")

    return FileResponse(
        path=output_path,
        filename="response_spectrum_result.csv",
        media_type="text/csv",
    )