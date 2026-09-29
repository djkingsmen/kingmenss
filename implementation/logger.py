"""
logger.py — High-Precision Execution Profiler & Log Persistence Subsystem
Paper: Quantum Information Entanglement Index for Facial Expression Analysis (ECML PKDD 2026)

Features:
- Sub-millisecond timing using time.perf_counter()
- Dual-target logging: Console output + Persisted files per run
- Timestamped log files saved in implementation/logs/run_YYYYMMDD_HHMMSS.log
- Structured JSON timing summaries saved in implementation/logs/run_YYYYMMDD_HHMMSS_timings.json
- Automatically maintained 'latest_run.log' and 'latest_timings.json'
- Platform, hardware, and python environment metadata capture
"""

import sys
import os
import time
import json
import platform
import datetime


class ExecutionLogger:
    def __init__(self, run_name: str = "qiei_verification"):
        self.run_name = run_name
        self.start_wall_time = datetime.datetime.now()
        self.start_perf = time.perf_counter()
        
        # Log directory: implementation/logs/
        self.log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        timestamp_str = self.start_wall_time.strftime("%Y%m%d_%H%M%S")
        self.log_file_path = os.path.join(self.log_dir, f"run_{timestamp_str}.log")
        self.json_file_path = os.path.join(self.log_dir, f"run_{timestamp_str}_timings.json")
        self.latest_log_path = os.path.join(self.log_dir, "latest_run.log")
        self.latest_json_path = os.path.join(self.log_dir, "latest_timings.json")
        
        self.stages = []
        self.current_stage = None
        self.subtasks = []
        
        # Open persistent file handle
        self._fh = open(self.log_file_path, "w", encoding="utf-8")
        
        # Initial header
        self.log_raw(f"================================================================================")
        self.log_raw(f"  QIEI EXECUTION & BENCHMARK PROFILE: {self.run_name.upper()}")
        self.log_raw(f"  Start Timestamp: {self.start_wall_time.isoformat()}")
        self.log_raw(f"  System: {platform.system()} {platform.release()} ({platform.machine()})")
        self.log_raw(f"  Processor: {platform.processor()}")
        self.log_raw(f"  Python Version: {platform.python_version()} ({platform.python_implementation()})")
        self.log_raw(f"  Log File:  {self.log_file_path}")
        self.log_raw(f"  JSON Data: {self.json_file_path}")
        self.log_raw(f"================================================================================\n")

    def log_raw(self, msg: str):
        """Writes to both console and file handle."""
        print(msg)
        if self._fh and not self._fh.closed:
            self._fh.write(msg + "\n")
            self._fh.flush()

    def info(self, msg: str):
        elapsed = time.perf_counter() - self.start_perf
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted = f"[{timestamp} | +{elapsed:7.3f}s] [INFO]  {msg}"
        self.log_raw(formatted)

    def success(self, msg: str):
        elapsed = time.perf_counter() - self.start_perf
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted = f"[{timestamp} | +{elapsed:7.3f}s] [PASS]  {msg}"
        self.log_raw(formatted)

    def warn(self, msg: str):
        elapsed = time.perf_counter() - self.start_perf
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted = f"[{timestamp} | +{elapsed:7.3f}s] [WARN]  {msg}"
        self.log_raw(formatted)

    def error(self, msg: str):
        elapsed = time.perf_counter() - self.start_perf
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted = f"[{timestamp} | +{elapsed:7.3f}s] [ERROR] {msg}"
        self.log_raw(formatted)

    def start_stage(self, stage_id: str, stage_title: str):
        """Begins tracking a named execution stage."""
        self.current_stage = {
            "id": stage_id,
            "title": stage_title,
            "start_time": time.perf_counter(),
            "start_wall": datetime.datetime.now().isoformat(),
            "subtasks": []
        }
        self.log_raw("\n" + "-" * 80)
        self.info(f"STARTING >> [{stage_id}] {stage_title}")
        self.log_raw("-" * 80)

    def end_stage(self, status: str = "PASSED", metadata: dict = None):
        """Ends tracking the current stage and computes duration."""
        if not self.current_stage:
            return
        end_time = time.perf_counter()
        duration = end_time - self.current_stage["start_time"]
        self.current_stage["end_time"] = end_time
        self.current_stage["duration_sec"] = duration
        self.current_stage["status"] = status
        self.current_stage["metadata"] = metadata or {}
        
        self.stages.append(self.current_stage)
        self.success(f"COMPLETED >> [{self.current_stage['id']}] {self.current_stage['title']} in {duration:.4f}s ({status})")
        self.current_stage = None

    def log_subtask(self, name: str, duration_sec: float, throughput_info: str = ""):
        """Records granular sub-task timing inside the active stage."""
        entry = {
            "name": name,
            "duration_sec": duration_sec,
            "throughput": throughput_info
        }
        if self.current_stage:
            self.current_stage["subtasks"].append(entry)
        thru_str = f" ({throughput_info})" if throughput_info else ""
        self.info(f"   * Subtask '{name}': {duration_sec * 1000.0:.2f} ms{thru_str}")

    def finalize(self) -> dict:
        """Closes files and persists JSON metrics report."""
        total_duration = time.perf_counter() - self.start_perf
        end_wall_time = datetime.datetime.now()
        
        self.log_raw("\n" + "=" * 80)
        self.log_raw("                       EXECUTION SUMMARY & VERIFICATION DASHBOARD")
        self.log_raw("=" * 80)
        self.log_raw(f"  {'Stage ID':<10} | {'Stage Title':<38} | {'Status':<8} | {'Duration':<10}")
        self.log_raw("  " + "-" * 74)
        
        for stg in self.stages:
            badge = "[PASS]" if stg["status"] == "PASSED" else "[FAIL]"
            dur_str = f"{stg['duration_sec']:.3f}s"
            self.log_raw(f"  {badge} {stg['id']:<4} | {stg['title']:<38} | {stg['status']:<8} | {dur_str:<10}")
            for sub in stg.get("subtasks", []):
                self.log_raw(f"       -> {sub['name']:<35} : {sub['duration_sec']*1000.0:6.1f} ms {sub.get('throughput', '')}")
                
        self.log_raw("  " + "-" * 74)
        self.log_raw(f"  Total Wall Clock Time: {total_duration:.3f} seconds")
        self.log_raw(f"  Timestamped Log Saved: {self.log_file_path}")
        self.log_raw(f"  JSON Benchmark Saved:  {self.json_file_path}")
        self.log_raw("=" * 80 + "\n")
        
        # Build JSON document
        payload = {
            "run_name": self.run_name,
            "start_time": self.start_wall_time.isoformat(),
            "end_time": end_wall_time.isoformat(),
            "total_duration_sec": total_duration,
            "environment": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version()
            },
            "stages": [
                {
                    "id": s["id"],
                    "title": s["title"],
                    "status": s["status"],
                    "duration_sec": s["duration_sec"],
                    "metadata": s.get("metadata", {}),
                    "subtasks": s.get("subtasks", [])
                }
                for s in self.stages
            ]
        }
        
        # Write timestamped JSON
        with open(self.json_file_path, "w", encoding="utf-8") as jf:
            json.dump(payload, jf, indent=2)
            
        # Write 'latest_run.log' copy
        try:
            with open(self.log_file_path, "r", encoding="utf-8") as src, open(self.latest_log_path, "w", encoding="utf-8") as dst:
                dst.write(src.read())
            with open(self.latest_json_path, "w", encoding="utf-8") as jf:
                json.dump(payload, jf, indent=2)
        except Exception:
            pass
            
        if self._fh and not self._fh.closed:
            self._fh.close()
            
        return payload
