import subprocess
import os
from tqdm import tqdm
from datetime import datetime

# 🛠 perf 바이너리 경로 (수정 필요 시 아래 경로 변경)
PERF_PATH = "/home/oxqnd/linux-5.15.137/tools/perf/perf"

# 📁 저장 경로 설정
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# 🕒 수집 설정
DURATION_SEC = 30
INTERVAL_MS = 100


def build_perf_cmd(output_path: str, mode: str) -> list:
    # attack은 강한 연산, normal은 상대적으로 가벼운 연산
    workload = (
        "for i in range(100000000): x = i*i%10000"
        if mode == "attack" else
        "for i in range(100000000): x = i*i%12345"
    )

    return [
        PERF_PATH, "stat",
        "-e", "cache-references,cache-misses,branches,branch-misses",
        "-I", str(INTERVAL_MS),
        "-o", output_path,
        "--", "python3", "-c", workload
    ]


def run_perf_capture(mode: str):
    assert mode in ['normal', 'attack']
    output_path = os.path.join(DATA_DIR, f"{mode}_perf.txt")
    cmd = build_perf_cmd(output_path, mode)

    print(f"\n[*] {mode.upper()} 상황에서 perf 수집 시작 → {output_path}")
    print("[*] 부하 코드 실행 중..." if mode == "attack" else "[*] 일반 연산 실행 중...")

    start_time = datetime.now()
    eta = (start_time.timestamp() + DURATION_SEC)
    eta_str = datetime.fromtimestamp(eta).strftime('%H:%M:%S')

    with tqdm(
        total=DURATION_SEC,
        desc=f"⏱ {mode.upper()} 수집 (예정 종료: {eta_str})",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ncols=90
    ) as pbar:
        proc = subprocess.Popen(cmd)
        try:
            # perf 수집이 끝날 때까지 기다림
            proc.wait(timeout=DURATION_SEC + 5)
        except subprocess.TimeoutExpired:
            print("[!] 수집 시간 초과: 강제 종료 시도")
            proc.terminate()
            proc.wait()

        # 수동으로 tqdm 100%로 설정
        pbar.n = DURATION_SEC
        pbar.refresh()

    print(f"[*] {mode.upper()} 수집 종료 시각: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    run_perf_capture("normal")
    run_perf_capture("attack")
    print("\n[✓] 모든 수집 완료. data/ 폴더를 확인하세요.")
