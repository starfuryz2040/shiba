import abc
import psutil
import time
import datetime

monitoring_active = False

class DMBSParent(metaclass=abc.ABCMeta):

    def __init__(self):
        self.log = []
        self.target_table = None
        self.run_num = None
        self.system = None
        self.udf_type = None

    def update_global(self, status):
        global monitoring_active
        monitoring_active = status
        print(f'monitoring active: {monitoring_active}')

    def monitor_system_resources(self, interval, query):
        print("Monitoring system resources...")

        while monitoring_active:
            now = datetime.datetime.now()
            cpu_stats = psutil.cpu_stats()
            interrupts = cpu_stats.interrupts
            ctx_switches = cpu_stats.ctx_switches
            soft_interrupts = cpu_stats.soft_interrupts

            cpu_percent = psutil.cpu_percent(interval=None)  # Non-blocking call
            cpu_times_percent = psutil.cpu_times_percent(interval=None)
            virtual_memory = psutil.virtual_memory()
            memory_percent = virtual_memory.percent
            time.sleep(interval)

            pid_list = []

            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'memory_info']):
                try:
                    # Get process information
                    pid = proc.info['pid']
                    name = proc.info['name']
                    cpu_percent_pid = proc.cpu_percent(interval=None)  # Calculate CPU usage over a short interval
                    memory_percent_pid = proc.info['memory_percent']
                    if cpu_percent_pid > 1 or memory_percent_pid > 1:
                        # memory_percent_pid = proc.info['memory_percent']
                        memory_info = proc.info['memory_info']
                        rss_mb = memory_info.rss / (1024 * 1024) if memory_info else 0  # Resident Set Size in MB

                        # print(f"{pid:<8} | {name:<30} | {cpu_percent_pid:<10.2f} | {memory_percent_pid:<10.2f} | {rss_mb:<15.2f}")

                        pid_entry = {"pid": pid,
                                     "name": name,
                                     "cpu_percent": cpu_percent_pid,
                                     "memory_percent": memory_percent_pid,
                                     "rss_mb": rss_mb
                                     }

                        pid_list.append(pid_entry)


                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    # Handle cases where a process might disappear or access is denied
                    continue

            system_entry = {
                "experiment" : self.target_table,
                "run" : self.run_num,
                "system" : self.system,
                "udf" : self.udf_type,
                "query": query,
                "time": now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                "cpu_percent": cpu_percent,
                "cpu_user_percent": cpu_times_percent[0],
                "cpu_nice_percent": cpu_times_percent[1],
                "cpu_system_percent": cpu_times_percent[2],
                "cpu_idle_percent": cpu_times_percent[3],
                "cpu_iowait1_percent": cpu_times_percent[4],
                "cpu_irq_percent": cpu_times_percent[5],
                "cpu_softirq_percent": cpu_times_percent[6],
                "cpu_steal_percent": cpu_times_percent[7],
                "cpu_guest_percent": cpu_times_percent[8],
                "cpu_guest_nice_percent": cpu_times_percent[9],
                "interrupts": interrupts,
                "ctx_switches": ctx_switches,
                "soft_interrupts": soft_interrupts,
                "memory_percent": memory_percent,
                "pids": pid_list

            }

            self.log.append(system_entry)

        print("Monitoring stopped.")
