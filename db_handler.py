import psycopg2
from psycopg2.extras import Json, RealDictCursor
import uuid
from datetime import datetime
import hashlib
import hmac
import os

# pbkdf2 密码安全哈希辅助函数
def hash_password(password: str) -> str:
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return salt.hex() + ":" + key.hex()

def verify_password(stored_password_hash: str, provided_password: str) -> bool:
    try:
        salt_hex, key_hex = stored_password_hash.split(":")
        salt = bytes.fromhex(salt_hex)
        key = bytes.fromhex(key_hex)
        new_key = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt, 100000)
        return hmac.compare_digest(new_key, key)
    except Exception:
        return False


class MSA_DB:
    def __init__(self):
        password = os.environ.get("POSTGRES_PASSWORD")
        if not password:
            raise RuntimeError("缺少必需环境变量 POSTGRES_PASSWORD")

        self.conn_params = {
            "dbname": os.environ.get("POSTGRES_DB", "msa_db"),
            "user": os.environ.get("POSTGRES_USER", "postgres"),
            "password": password,
            "host": os.environ.get("POSTGRES_HOST", "postgres"),
            "port": int(os.environ.get("POSTGRES_PORT", "5432")),
        }
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        conn = psycopg2.connect(**self.conn_params)
        cur = conn.cursor()

        # 1. 创建用户表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 2. 创建会话表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id VARCHAR(50) PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 兼容性升级：若 sessions 表之前存在但没有 user_id 外键，则安全注入
        cur.execute("""
            ALTER TABLE sessions ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE
        """)

        # 3. 创建消息历史表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(50) REFERENCES sessions(session_id) ON DELETE CASCADE,
                role VARCHAR(20),  -- 'user' 或 'assistant'
                content TEXT,
                emotion_score FLOAT,    -- 仅对 user 消息有效
                emotion_polarity VARCHAR(20), -- 仅对 user 消息有效
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 兼容性升级：为 messages 表注入 video_url 字段，存储多模态视频路径
        cur.execute("""
            ALTER TABLE messages ADD COLUMN IF NOT EXISTS video_url VARCHAR(255)
        """)

        conn.commit()
        cur.close()
        conn.close()

    # --- 身份认证与用户管理 ---
    def register_user(self, username, password):
        """注册新用户"""
        conn = psycopg2.connect(**self.conn_params)
        cur = conn.cursor()
        try:
            p_hash = hash_password(password)
            cur.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING user_id",
                (username, p_hash)
            )
            user_id = cur.fetchone()[0]
            conn.commit()
            return user_id
        except Exception as e:
            conn.rollback()
            print(f"用户注册失败: {e}")
            return None
        finally:
            cur.close()
            conn.close()

    def verify_user(self, username, password):
        """验证用户登录并返回用户信息"""
        conn = psycopg2.connect(**self.conn_params)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        try:
            cur.execute("SELECT user_id, username, password_hash FROM users WHERE username = %s", (username,))
            user = cur.fetchone()
            if user and verify_password(user["password_hash"], password):
                return {"user_id": user["user_id"], "username": user["username"]}
            return None
        except Exception as e:
            print(f"用户校验发生故障: {e}")
            return None
        finally:
            cur.close()
            conn.close()

    # --- 多会话与会话归属管理 ---
    def create_session(self, user_id=None):
        """绑定指定的用户，创建一个新的对话 Session"""
        session_id = str(uuid.uuid4())
        conn = psycopg2.connect(**self.conn_params)
        cur = conn.cursor()
        cur.execute("INSERT INTO sessions (session_id, user_id) VALUES (%s, %s)", (session_id, user_id))
        conn.commit()
        cur.close()
        conn.close()
        return session_id

    def get_user_sessions(self, user_id):
        """拉取该用户下所有的历史会话"""
        conn = psycopg2.connect(**self.conn_params)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT session_id, created_at FROM sessions WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,)
        )
        sessions = cur.fetchall()
        cur.close()
        conn.close()
        return sessions

    def delete_session(self, session_id):
        """硬删除指定会话"""
        conn = psycopg2.connect(**self.conn_params)
        cur = conn.cursor()
        try:
            cur.execute("DELETE FROM messages WHERE session_id = %s", (session_id,))
            cur.execute("DELETE FROM sessions WHERE session_id = %s", (session_id,))
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            print(f"删除会话失败: {e}")
            return False
        finally:
            cur.close()
            conn.close()

    # 新增对话记录插入接口 - 支持情绪分析结果和多模态视频信息的存储
    def add_message(self, session_id, role, content, score=None, polarity=None, video_url=None):
        """插入一条对话记录"""
        conn = psycopg2.connect(**self.conn_params)
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO messages (session_id, role, content, emotion_score, emotion_polarity, video_url)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (session_id, role, content, score, polarity, video_url)
        )
        conn.commit()
        cur.close()
        conn.close()

    # 历史数据查询 - 支持按时间倒序拉取最近的上下文历史，并兼容多模态视频信息
    def get_session_history(self, session_id, limit=10):
        """获取最近的上下文历史"""
        conn = psycopg2.connect(**self.conn_params)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            SELECT role, content, video_url
            FROM (SELECT role, content, video_url, created_at
                  FROM messages
                  WHERE session_id = %s
                  ORDER BY created_at DESC
                      LIMIT %s) sub
            ORDER BY created_at ASC
            """,
            (session_id, limit)
        )
        history = cur.fetchall()
        cur.close()
        conn.close()
        return history
    
    def get_emotion_history(self, session_id):
        """获取某会话下所有用户发言的情绪强度得分"""
        conn = psycopg2.connect(**self.conn_params)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        try:
            cur.execute(
                """
                SELECT created_at, emotion_score, emotion_polarity
                FROM messages
                WHERE session_id = %s AND role = 'user' AND emotion_score IS NOT NULL
                ORDER BY created_at ASC
                """,
                (session_id,)
            )
            history = cur.fetchall()
            return history
        except Exception as e:
            print(f"查询历史情绪数据失败: {e}")
            return []
        finally:
            cur.close()
            conn.close()
