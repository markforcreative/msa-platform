from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles  # 【新增】引入静态资源挂载
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
from pydantic import BaseModel
import shutil
import os
import uuid
import httpx
import json
import re

# 导入底层引擎与数据库处理器
from emotion_engine import EmotionEngine
from db_handler import MSA_DB

ml_models = {}

# ================= 配置区 =================
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
ONNX_MODEL_PATH = os.environ.get("ONNX_MODEL_PATH", "lnln_deploy.onnx")
MAX_UPLOAD_SIZE_MB = int(os.environ.get("MAX_UPLOAD_SIZE_MB", "200"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
# ==========================================

# StaticFiles 在应用导入阶段会校验目录，因此必须在挂载前创建。
os.makedirs(UPLOAD_DIR, exist_ok=True)

class UserAuth(BaseModel):
    username: str
    password: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 确保视频存储目录存在
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    try:
        print("[Lifespan] 正在初始化多模态情感分析引擎...")
        engine = EmotionEngine(onnx_path=ONNX_MODEL_PATH)
        ml_models["engine"] = engine
        
        print("[Lifespan] 正在初始化 PostgreSQL 数据库连接...")
        db = MSA_DB()
        ml_models["db"] = db
        
    except Exception as e:
        print(f"❌ 引擎或数据库初始化失败！报错信息：{e}")
        raise e
        
    yield
    print("🛑 正在关闭服务器并清理模型内存...")
    ml_models.clear()

app = FastAPI(title="MSA 多模态智能咨询服务端", lifespan=lifespan)

# 将 UPLOAD_DIR (uploads 文件夹) 挂载至公网访问路由 /uploads
# 真机和模拟器可通过 https://<sakurfrp-host>:<port>/uploads/<filename> 访问视频
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.get("/health")
async def health():
    ready = "engine" in ml_models and "db" in ml_models
    if not ready:
        raise HTTPException(status_code=503, detail="核心服务尚未就绪")
    return {"status": "ok"}


# ================= 身份认证路由 =================
@app.post("/auth/register")
def register(user: UserAuth):
    if "db" not in ml_models:
        raise HTTPException(status_code=500, detail="数据库未正常连接")
    db = ml_models["db"]
    user_id = db.register_user(user.username, user.password)
    if not user_id:
        raise HTTPException(status_code=400, detail="注册失败，用户名可能已存在")
    return {"status": "success", "user_id": user_id, "username": user.username}

@app.post("/auth/login")
def login(user: UserAuth):
    if "db" not in ml_models:
        raise HTTPException(status_code=500, detail="数据库未正常连接")
    db = ml_models["db"]
    user_info = db.verify_user(user.username, user.password)
    if not user_info:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    return {"status": "success", "user_id": user_info["user_id"], "username": user_info["username"]}


# ================= 多会话管理路由 =================
@app.post("/sessions/create")
def create_session(user_id: int = Form(None)):
    """新建会话，支持关联 user_id"""
    if "db" not in ml_models:
        raise HTTPException(status_code=500, detail="数据库未正常连接")
    db = ml_models["db"]
    session_id = db.create_session(user_id=user_id)
    return {"status": "success", "session_id": session_id}

@app.get("/users/{user_id}/sessions")
def get_user_sessions_api(user_id: int):
    """获取用户关联的所有会话"""
    if "db" not in ml_models:
        raise HTTPException(status_code=500, detail="数据库未正常连接")
    db = ml_models["db"]
    try:
        sessions = db.get_user_sessions(user_id)
        formatted_sessions = []
        for s in sessions:
            formatted_sessions.append({
                "session_id": s["session_id"],
                "created_at": s["created_at"].strftime("%Y-%m-%d %H:%M:%S") if s["created_at"] else ""
            })
        return {"status": "success", "sessions": formatted_sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}")
def delete_session_api(session_id: str):
    """硬删除指定会话"""
    if "db" not in ml_models:
        raise HTTPException(status_code=500, detail="数据库未运行")
    db = ml_models["db"]
    success = db.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=500, detail="删除会话失败")
    return {"status": "success"}


# ================= 对话流式流处理器 =================
# 接收持久化的 video_path 以及可暴露的相对路径 relative_video_url
async def chat_stream_generator(session_id: str, text: str = None, video_path: str = None, relative_video_url: str = None):
    db = ml_models.get("db")
    analyzer = ml_models.get("engine")
    
    score, polarity = None, None
    user_typed_text = text if text else ""
    user_text = user_typed_text

    try:
        if video_path and os.path.exists(video_path):
            # 将持久化保存于 uploads/ 的文件路径传递给情感分析工具，保证不因中途删除而出错
            transcribed_text, score, polarity = await run_in_threadpool(
                analyzer.process_video, video_path
            )

            has_valid_voice = (
                transcribed_text 
                and transcribed_text.strip() 
                and transcribed_text.strip() != "啊"
            )

            if user_typed_text.strip():
                if has_valid_voice:
                    user_text = (
                        f"{user_typed_text.strip()}\n\n"
                        f"<span style='display: block; margin-top: 8px; line-height: 1.4; font-size: 0.85em; font-style: italic; color: inherit; opacity: 0.65;'>"
                        f"🎙️ 视频语音: {transcribed_text.strip()}"
                        f"</span>"
                    )
                else:
                    user_text = user_typed_text.strip()
            else:
                user_text = transcribed_text if transcribed_text else "（未检测到有效输入）"
        
        if not user_text.strip():
            user_text = "（未检测到有效输入）"

        # 写入数据库时，保存相对的静态文件访问路径：/uploads/xxx.mp4
        await run_in_threadpool(
            db.add_message, session_id, "user", user_text, score, polarity, relative_video_url
        )

        history = await run_in_threadpool(db.get_session_history, session_id, 12)

        system_prompt = """你是一位专业且充满共情力的心理咨询师。
你的主要沟通方式是文本聊天。当有额外的多模态参数提供时，请综合考虑。

[交互原则]
1. 连贯性与记忆：请参考刚才的对话历史，留意用户话题的变化。若用户情绪发生大转折，请温柔地询问原因。
2. 回复自然：像真人朋友一样对话，不要像机器。绝对不要在回复中直接提及“多模态参数”、“极性”、“得分”等数据词汇。
3. 文本为主：如果当前没有多模态参数接入，请基于语义和对话历史进行共情。"""

        if score is not None and polarity is not None:
            system_prompt += f"""\n\n[当前句子的多模态底层感知 (仅作内部参考)]
- 实时情绪极性: {polarity}
- 情绪强度得分: {score} (-1.0 至 1.0)
(注: 这个参数表明了用户此刻说话时的真实状态。如果有强烈的极性负分，请务必增强安抚。)"""

            print(f"[{session_id[:8]}] 注入情绪特征 - 极性: {polarity} | 得分: {score:.2f}")

        messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            messages.append({"role": msg['role'], "content": msg['content']})

        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": True
        }

        full_response = ""
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", OLLAMA_URL, json=payload) as response:
                if response.status_code != 200:
                    yield f"❌ LLM 服务请求失败: Code {response.status_code}"
                    return
                
                async for line in response.aiter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if "message" in chunk and "content" in chunk["message"]:
                                content = chunk["message"]["content"]
                                full_response += content
                                yield content
                        except Exception as parse_err:
                            print(f"解析 chunk 失败: {parse_err}")

        await run_in_threadpool(
            db.add_message, session_id, "assistant", full_response
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield f"\n❌ 服务端处理异常: {str(e)}"
        
    finally:
        # 【改动】不再在此处删除临时上传的视频。
        # 视频将被保留在 uploads/ 目录中，以便移动端历史列表静默刷新后仍可继续拉取播放。
        pass


@app.get("/sessions/{session_id}/history")
def get_session_history_api(session_id: str):
    if "db" not in ml_models:
        raise HTTPException(status_code=500, detail="数据库未正常连接")
    db = ml_models["db"]
    try:
        history = db.get_session_history(session_id, limit=30)
        return {"status": "success", "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/emotions")
def get_session_emotions_api(session_id: str):
    if "db" not in ml_models:
        raise HTTPException(status_code=500, detail="数据库未正常连接")
    db = ml_models["db"]
    try:
        data = db.get_emotion_history(session_id)
        formatted_data = []
        for row in data:
            formatted_data.append({
                "time": row["created_at"].strftime("%H:%M:%S") if row["created_at"] else "",
                "score": row["emotion_score"],
                "polarity": row["emotion_polarity"]
            })
        return {"status": "success", "data": formatted_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/chat")
async def chat(
    session_id: str,
    text: str = Form(None),
    video_file: UploadFile = File(None)
):
    if "engine" not in ml_models or "db" not in ml_models:
        raise HTTPException(status_code=500, detail="核心引擎或数据库未在线")

    saved_video_path = None
    relative_video_url = None

    if video_file:
        if video_file.content_type and not video_file.content_type.startswith("video/"):
            raise HTTPException(status_code=415, detail="只允许上传视频文件")

        file_id = str(uuid.uuid4())[:8]
        original_name = (video_file.filename or "upload.bin").replace("\\", "/").split("/")[-1]
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", original_name)[:128] or "upload.bin"
        filename = f"{file_id}_{safe_name}"
        
        # 将接收的文件持续化保存到挂载的静态文件夹 uploads 中
        saved_video_path = os.path.join(UPLOAD_DIR, filename)
        bytes_written = 0
        try:
            with open(saved_video_path, "wb") as buffer:
                while chunk := await video_file.read(1024 * 1024):
                    bytes_written += len(chunk)
                    if bytes_written > MAX_UPLOAD_SIZE_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"上传文件不能超过 {MAX_UPLOAD_SIZE_MB} MB",
                        )
                    buffer.write(chunk)
        except Exception:
            if os.path.exists(saved_video_path):
                os.remove(saved_video_path)
            raise
        finally:
            await video_file.close()
        
        # 定义暴露出去的相对资源 URL
        relative_video_url = f"/uploads/{filename}"

    return StreamingResponse(
        chat_stream_generator(
            session_id, 
            text=text, 
            video_path=saved_video_path, 
            relative_video_url=relative_video_url
        ),
        media_type="text/event-stream"
    )


@app.get("/")
async def root():
    return {"message": "多模态心理咨询后端服务已启动。"} 
