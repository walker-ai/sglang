import logging
import multiprocessing
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from enum import Enum
from logging.handlers import RotatingFileHandler

import prctl
import psutil
import requests

sglang_agent_log_path = f'/home/admin/logs/sglang_agent/'
glogger = logging.getLogger(__file__)


def get_local_ip():
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    return ip


def get_local_host():
    return socket.gethostname()


def setup_logging(logfile=None, console_min_level=logging.INFO, file_min_level=logging.DEBUG):
    """ setup logging for test, output logs on console (default logging.WARN) and logfile(default logging.DEBUG)"""
    if logfile is None:
        filename = os.path.basename(sys.argv[0]).replace(".py", ".log")
    else:
        filename = logfile
    logger = logging.getLogger(__file__)
    fmter = logging.Formatter(f'[{get_local_ip()}]%(asctime)s %(process)d %(levelname)s %(message)s',
                              datefmt='%a, %d %b %Y %H:%M:%S')
    hdlr = RotatingFileHandler(filename, maxBytes=100 * 1024 * 1024, backupCount=30)
    hdlr.setLevel(file_min_level)
    hdlr.setFormatter(fmter)
    logger.addHandler(hdlr)
    # console_hdlr = logging.StreamHandler()
    # console_hdlr.setLevel(logging.INFO)
    # fmter = logging.Formatter('%(message)s')
    # console_hdlr.setFormatter(fmter)
    # logger.addHandler(console_hdlr)
    logger.setLevel(logging.INFO)


os.makedirs(sglang_agent_log_path, exist_ok=True)
setup_logging(os.path.join(sglang_agent_log_path, 'agent.log'))


def agent_watchdog(agent_proc, target_pid):
    def watchdog(agent_pid, target_pid):
        try:
            agent_proc.join()
        except Exception as e:
            glogger.info('agent_watchdog, agent_pid:%s, kill myself: %s', agent_pid, target_pid)
            os.system(f'kill {target_pid}')

    thread = threading.Thread(target=watchdog, args=(agent_proc, target_pid,), daemon=True)
    thread.start()


def try_wait_sglang_server(model_id, port=8189, lock_port=12332, timeout=1200, **kwargs) -> bool:
    sub_counts, subdirs = count_files_and_dirs(model_id)
    if sub_counts == 1:
        # 有可能这个包解压后是个子目录，所有大模型文件在子目录下
        model_id = os.path.join(model_id, subdirs[0])
    kwargs['model_path'] = model_id
    kwargs['port'] = port
    kwargs['parent_pid'] = os.getpid()
    sglang_agent = SGlangAgent(port, lock_port)
    sglang_proc = multiprocessing.Process(
        target=sglang_agent.process, kwargs=kwargs, daemon=True)
    sglang_proc.start()
    timeout_ts = time.time() + timeout
    while time.time() < timeout_ts:
        launcher_state = sglang_agent.get_server_state()
        if launcher_state == ServerState.started:
            glogger.info('launcher start success.')
            agent_watchdog(sglang_proc, os.getpid())
            return True
        else:
            time.sleep(2)
            glogger.info('wait launcher start.')
    glogger.error('launcher start timeout.')
    return False


def count_files_and_dirs(directory):
    num_files = 0
    num_dirs = 0
    subdirs = []
    for root, dirs, files in os.walk(directory):
        glogger.info('walk %s %s', dirs, files)
        num_dirs += len(dirs)
        num_files += len(files)
        subdirs = dirs
        break
    return num_files + num_dirs, subdirs


def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in children:
        process.send_signal(sig)


def is_parent_alive(parent_pid):
    try:
        psutil.Process(parent_pid)
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False
    except:
        return True


def popen_exec(cmd):
    proc = subprocess.run(cmd, shell=True)
    return_code = proc.returncode
    if return_code != 0:
        raise Exception(
            f"return_value: {return_code}, failed executing command '{cmd}'"
        )


def check_process_exists(command):
    for process in psutil.process_iter(['pid', 'cmdline']):
        if process.info['cmdline'] and command in process.info['cmdline']:
            return True
    return False


def check_port_open(server_ip, server_port):
    ip_port = (server_ip, server_port)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)
    return s.connect_ex(ip_port) == 0


def sglang_health_check(server_ip, server_port):
    health_check_url = f'http://{server_ip}:{server_port}/health'
    response = requests.get(health_check_url)
    return response.status_code == 200


class Singleton(object):
    def __init__(self, lock_port, parent_pid):
        self.lock = False
        self.lock_port = lock_port
        self.parent_pid = parent_pid

    def __enter__(self):
        while is_parent_alive(self.parent_pid):
            try:
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._socket.bind(('localhost', self.lock_port))
                self._socket.listen(1)
                self.lock = True
                return self
            except socket.error:
                # glogger.warning('lock failed, sleep and try again.')
                time.sleep(3)
            except Exception as e:
                glogger.error('socket singleton exception:%s', e)
                raise e

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.lock:
            self._socket.close()


class ServerState(Enum):
    not_exist = 1
    initting = 2
    started = 3


class SGlangAgent:
    server_cmd = "sglang.launch_server"

    def __init__(self, port: int = 8189, lock_port: int = 12332):
        self.port = port
        self.lock_port = lock_port

    def start_server(self, **kwargs):
        glogger.info(f'try to start sglang server with args: {kwargs}')
        # 杀死子进程
        kill_child_processes(os.getpid())
        server_args = ''
        for k, v in kwargs.items():
            k = k.replace('_', '-')
            server_args += f'--{k} {v} '
        popen_exec(
            f'export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 && export RAY_memory_monitor_refresh_ms=0 && export PYTHONUSERBASE=intentionally-disabled && python -m {self.server_cmd} --host 127.0.0.1 {server_args} 2>&1 &')

    def get_server_state(self) -> ServerState:
        # 进程是否存在检查
        processs_exist = check_process_exists(self.server_cmd)
        glogger.info(f'processs_exist::{processs_exist}')
        if not processs_exist:
            return ServerState.not_exist

        local_ip = "127.0.0.1"
        # # 进程存在且端口已经打开，则认为已经启动成功
        if check_port_open(local_ip, self.port):
            if sglang_health_check(local_ip, self.port):
                return ServerState.started
            else:
                return ServerState.not_exist
        # 进程存在但端口未打开，则认为server在启动中
        else:
            return ServerState.initting

    def process(self, **kwargs):
        prctl.set_pdeathsig(signal.SIGHUP)
        prctl.set_pdeathsig(signal.SIGKILL)
        prctl.set_pdeathsig(signal.SIGTERM)
        prctl.set_pdeathsig(signal.SIGABRT)
        prctl.set_pdeathsig(signal.SIGINT)
        parent_pid = kwargs.pop('parent_pid')
        with Singleton(self.lock_port, parent_pid):
            try:
                while is_parent_alive(parent_pid):
                    try:
                        state = self.get_server_state()
                        glogger.info('launcher_state is:%s', state)
                        if state == ServerState.not_exist:
                            # 进程不存在，启动server
                            self.start_server(**kwargs)
                        elif state == ServerState.initting:
                            # 启动中
                            glogger.info('wait server')
                        else:
                            # 已经启动
                            glogger.info('started')
                        time.sleep(3)
                    except Exception as e:
                        glogger.info('agent process exception:%s', e)
            finally:
                kill_child_processes(os.getpid())
