import socket
import threading
import datetime
import argparse
import sys


# ============================================================
# 配置
# ============================================================

ENCODING = "utf-8"
BUFFER_SIZE = 4096

# 不再使用 Flask 常见的 5000 端口
DEFAULT_PORT = 9001


# ============================================================
# 工具函数
# ============================================================

def get_time():
    return datetime.datetime.now().strftime("%H:%M:%S")


def print_message(sender, message):
    """
    统一显示消息
    """
    print(f"\n[{get_time()}] {sender}: {message}")
    print("我: ", end="", flush=True)


# ============================================================
# TCP 聊天类
# ============================================================

class ChatClient:

    def __init__(self, conn, address=None):

        self.conn = conn
        self.address = address

        self.running = True

        # 防止多个线程同时发送数据
        self.send_lock = threading.Lock()

    # --------------------------------------------------------
    # 接收消息
    # --------------------------------------------------------

    def receive_messages(self):

        buffer = b""

        while self.running:

            try:

                data = self.conn.recv(BUFFER_SIZE)

                # 对方关闭连接
                if not data:

                    print("\n")
                    print("=" * 50)
                    print("对方已断开连接。")
                    print("=" * 50)

                    self.running = False
                    break

                # 加入缓冲区
                buffer += data

                # 一个 TCP 包可能包含多条消息
                while b"\n" in buffer:

                    line, buffer = buffer.split(b"\n", 1)

                    if not line:
                        continue

                    try:
                        message = line.decode(
                            ENCODING,
                            errors="replace"
                        )

                    except Exception:
                        message = str(line)

                    print_message("对方", message)

            except ConnectionResetError:

                print("\n对方强制关闭连接。")

                self.running = False
                break

            except OSError as e:

                if self.running:
                    print(f"\n接收消息失败：{e}")

                self.running = False
                break

            except Exception as e:

                if self.running:
                    print(f"\n接收消息失败：{e}")

                self.running = False
                break

    # --------------------------------------------------------
    # 发送消息
    # --------------------------------------------------------

    def send_message(self, message):

        try:

            # 添加消息结束符
            data = (message + "\n").encode(ENCODING)

            with self.send_lock:

                self.conn.sendall(data)

            print(f"[{get_time()}] 我: {message}")

        except BrokenPipeError:

            print("\n发送失败：对方已经断开连接。")
            self.running = False

        except ConnectionResetError:

            print("\n发送失败：连接被对方重置。")
            self.running = False

        except Exception as e:

            print(f"\n发送失败：{e}")
            self.running = False

    # --------------------------------------------------------
    # 启动聊天
    # --------------------------------------------------------

    def start(self):

        receive_thread = threading.Thread(
            target=self.receive_messages,
            daemon=True
        )

        receive_thread.start()

        print()
        print("=" * 50)
        print("                  LAN CHAT")
        print("=" * 50)
        print("TCP 连接成功。")
        print(f"编码：{ENCODING}")
        print("输入文字后按 Enter 发送。")
        print("输入 /exit 退出。")
        print("=" * 50)

        while self.running:

            try:

                message = input("我: ")

                # 退出
                if message.strip() == "/exit":

                    print("正在关闭连接...")

                    self.running = False
                    break

                # 空消息不发送
                if not message.strip():
                    continue

                self.send_message(message)

            except KeyboardInterrupt:

                print("\n正在关闭连接...")

                self.running = False
                break

            except EOFError:

                self.running = False
                break

        self.close()

    # --------------------------------------------------------
    # 关闭连接
    # --------------------------------------------------------

    def close(self):

        self.running = False

        try:
            self.conn.shutdown(socket.SHUT_RDWR)
        except:
            pass

        try:
            self.conn.close()
        except:
            pass


# ============================================================
# Server
# ============================================================

def run_server(port):

    server_socket = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    )

    # 允许端口快速重新使用
    server_socket.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1
    )

    try:

        server_socket.bind(
            ("0.0.0.0", port)
        )

        server_socket.listen(5)

    except Exception as e:

        print()
        print("服务器启动失败。")
        print(f"错误：{e}")
        print()
        print(f"可能是端口 {port} 已经被其他程序占用。")
        print("可以尝试：")
        print("python chat.py server --port 9002")
        return

    print()
    print("=" * 50)
    print("              LAN CHAT SERVER")
    print("=" * 50)
    print(f"监听地址：0.0.0.0")
    print(f"监听端口：{port}")
    print()
    print("等待另一台电脑连接...")
    print("=" * 50)

    try:

        while True:

            try:

                conn, address = server_socket.accept()

                print()
                print("=" * 50)
                print("客户端连接成功")
                print(f"IP：{address[0]}")
                print(f"端口：{address[1]}")
                print("=" * 50)

                chat = ChatClient(
                    conn,
                    address
                )

                chat.start()

                print()
                print("当前连接已经关闭。")
                print("服务器继续等待新的连接...")

            except KeyboardInterrupt:

                print()
                print("服务器正在关闭...")
                break

            except Exception as e:

                print()
                print(f"连接处理失败：{e}")
                print("继续等待连接...")

    finally:

        try:
            server_socket.close()
        except:
            pass

        print("服务器已关闭。")


# ============================================================
# Client
# ============================================================

def run_client(ip, port):

    print()
    print("=" * 50)
    print("              LAN CHAT CLIENT")
    print("=" * 50)

    print(f"目标 IP：{ip}")
    print(f"目标端口：{port}")
    print()
    print("正在建立 TCP 连接...")

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    )

    # TCP 连接超时
    sock.settimeout(10)

    try:

        sock.connect(
            (ip, port)
        )

        # 恢复阻塞模式
        sock.settimeout(None)

    except socket.timeout:

        print()
        print("连接超时。")
        print()
        print("请检查：")
        print("1. 对方电脑是否启动 server")
        print("2. IP 地址是否正确")
        print(f"3. 端口 {port} 是否正确")
        print("4. Windows 防火墙是否允许 Python")
        sock.close()
        return

    except ConnectionRefusedError:

        print()
        print("连接被拒绝。")
        print()
        print("请检查：")
        print("1. 对方电脑是否启动 server")
        print(f"2. 对方是否监听 TCP {port}")
        print("3. IP 地址是否正确")
        sock.close()
        return

    except Exception as e:

        print()
        print(f"连接失败：{e}")
        sock.close()
        return

    print()
    print("TCP 连接成功！")

    chat = ChatClient(sock)

    chat.start()


# ============================================================
# 主函数
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="局域网 TCP 文字聊天程序"
    )

    parser.add_argument(
        "mode",
        choices=["server", "client"],
        help="运行模式：server 或 client"
    )

    parser.add_argument(
        "--ip",
        default=None,
        help="服务器 IP 地址"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"TCP 端口，默认 {DEFAULT_PORT}"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Server
    # --------------------------------------------------------

    if args.mode == "server":

        run_server(
            args.port
        )

    # --------------------------------------------------------
    # Client
    # --------------------------------------------------------

    elif args.mode == "client":

        if args.ip is None:

            print()
            print("错误：client 模式需要指定服务器 IP。")
            print()
            print("例如：")
            print(
                "python chat.py client --ip 192.168.1.116"
            )

            return

        run_client(
            args.ip,
            args.port
        )


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":

    main()