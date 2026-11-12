#!/usr/bin/env python3
"""
Penerima data blok BDT lewat UDP (port 8800).

Mendukung dua format paket dari firmware:
  KBS4 : KBS4 <count> <diam/gerak> <eulerX> <eulerY> <eulerZ> <A0> <A1> <A2> KBX   (tanpa ID blok)
  KBS5 : KBS5 <ID> <count> <sisi> <state> <stabil> KBX                              (dengan ID blok)
         KBS5 <count> <sisi> <state> <stabil> KBX                                   (versi tanpa ID)

Blok dikenali dari ID (jika ada), kalau tidak dari alamat IP pengirim.

Pemakaian:
  python bdt_receiver.py                       tampilan tabel langsung
  python bdt_receiver.py --log sesi1.csv       sekaligus simpan semua paket ke CSV
  python bdt_receiver.py --raw                 cetak paket mentah (untuk debug)
  python bdt_receiver.py --port 8800

Tidak butuh library tambahan (hanya pustaka standar Python 3.8+).
Berhenti dengan Ctrl+C.
"""
import argparse
import csv
import os
import select
import socket
import sys
import time
from collections import deque

try:
    import msvcrt            # Windows: baca tombol tanpa perlu Enter
except ImportError:
    msvcrt = None

FACE_NAMES = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
OFFLINE_S = 3.0          # blok dianggap OFFLINE jika tidak ada paket selama ini (detik)
RENDER_PERIOD_S = 0.5    # seberapa sering tabel diperbarui
CMD_PORT = 8801          # port perintah di blok (kalibrasi)
DIAG_ORDER = [6, 3, 4, 5]
CAL_RESULT = {
    0: "OK (sisi diagonal dikalibrasi sebagai state 6)",
    1: "GAGAL: blok miring",
    2: "GAGAL: sisi atas bukan diagonal",
    3: "GAGAL: blok masih bergerak",
}


def face_name(f):
    return FACE_NAMES[f] if f is not None and 0 <= f < 6 else "--"


def read_key():
    """Baca satu tombol jika ada (Windows langsung, Linux/Mac perlu Enter). None jika tidak ada."""
    if msvcrt:
        if msvcrt.kbhit():
            return msvcrt.getwch()
        return None
    try:
        if select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline().strip()
            return line[:1] if line else None
    except (OSError, ValueError):
        pass
    return None


def send_cal(sock, blocks, only_id=None):
    """Kirim perintah kalibrasi ke blok yang dikenal (semua, atau hanya satu ID). Mengembalikan daftar nama blok."""
    sent = []
    for key, b in sorted(blocks.items()):
        if not key.startswith("B") or not b.ip:
            continue            # blok lama tanpa ID tidak mengerti perintah
        if only_id is not None and key != f"B{only_id}":
            continue
        sock.sendto(b"CAL", (b.ip, CMD_PORT))
        sent.append(key)
    return sent


class Block:
    def __init__(self, key):
        self.key = key
        self.ip = ""
        self.last_seen = 0.0
        self.rx = 0
        self.lost = 0
        self.restarts = 0
        self.last_restart = 0.0
        self.last_count = {}       # counter terakhir per jenis paket
        self.motion = None         # 0 diam, 1 bergerak (dari KBS4)
        self.euler = None
        self.analog = None
        self.state = None          # (sisi, state) terakhir yang valid (dari KBS5)
        self.stable = None
        self.rx_prev = 0
        self.t_prev = time.time()
        self.rate = 0.0
        self.flips = 0
        self.rots = 0
        self.rot_deg = 0
        self.ignore_until = 0.0     # abaikan perubahan sesaat setelah kalibrasi


def parse(line):
    """Ubah satu paket teks menjadi dict. Mengembalikan None jika formatnya tidak dikenal."""
    t = line.strip("\x00\r\n ").split("\t")
    if len(t) < 3 or t[-1] != "KBX":
        return None
    try:
        if t[0] == "KBACK" and len(t) == 5:     # balasan perintah kalibrasi
            return {"type": "KBACK", "id": int(t[1]), "result": int(t[2]), "face": int(t[3])}
        if t[0] == "KBS4" and len(t) == 10:
            return {
                "type": "KBS4", "id": None, "count": int(t[1]), "motion": int(t[2]),
                "euler": (float(t[3]), float(t[4]), float(t[5])),
                "analog": (float(t[6]), float(t[7]), float(t[8])),
            }
        if t[0] == "KBS5" and len(t) == 7:      # dengan ID
            return {"type": "KBS5", "id": int(t[1]), "count": int(t[2]),
                    "face": int(t[3]), "code": int(t[4]), "stable": int(t[5])}
        if t[0] == "KBS5" and len(t) == 6:      # tanpa ID
            return {"type": "KBS5", "id": None, "count": int(t[1]),
                    "face": int(t[2]), "code": int(t[3]), "stable": int(t[4])}
    except ValueError:
        return None
    return None


def render(blocks, events, now, port, log_name):
    out = []
    out.append(f"Penerima blok BDT | UDP port {port} | Ctrl+C untuk berhenti"
               + (f" | log: {log_name}" if log_name else ""))
    out.append("Tombol: c = kalibrasi diagonal SEMUA blok | 1-9 = hanya blok itu "
               "(sisi diagonal di atas, posisi state 6, blok diam)")
    out.append("")
    out.append(f"{'Blok':<14}{'IP':<16}{'Status':<9}{'Hz':>5}{'Rx':>8}{'Hilang':>8}"
               f"{'Reset':>6}{'State':>7}{'Sisi':>6}{'Flip':>6}{'Rot':>6}{'Der':>6}{'Diam?':>7}   Euler X / Y / Z")
    out.append("-" * 124)
    if not blocks:
        out.append("(belum ada paket masuk. Nyalakan blok, tunggu 10-15 detik sampai tersambung Wi-Fi)")
    for key, b in sorted(blocks.items()):
        online = (now - b.last_seen) < OFFLINE_S
        dt = now - b.t_prev
        if dt >= 1.0:
            b.rate = (b.rx - b.rx_prev) / dt
            b.rx_prev, b.t_prev = b.rx, now
        if not online:
            b.rate = 0.0
        state = str(b.state[1]) if b.state else "-"
        face = face_name(b.state[0]) if b.state else "--"
        if b.stable is not None:
            diam = "ya" if b.stable else "tidak"
        elif b.motion is not None:
            diam = "ya" if b.motion == 0 else "tidak"
        else:
            diam = "-"
        eul = "  ".join(f"{v:7.1f}" for v in b.euler) if b.euler else "-"
        out.append(f"{key:<14}{b.ip:<16}{'ONLINE' if online else 'OFFLINE':<9}{b.rate:>5.1f}"
           f"{b.rx:>8}{b.lost:>8}{b.restarts:>6}{state:>7}{face:>6}"
           f"{b.flips:>6}{b.rots:>5}{b.rot_deg:>6}{diam:>7}   {eul}")
    out.append("")
    out.append("Kejadian terakhir:")
    if events:
        out.extend("  " + e for e in events)
    else:
        out.append("  (belum ada)")
    print("\033[H\033[J" + "\n".join(out), flush=True)


def main():
    ap = argparse.ArgumentParser(description="Penerima UDP blok BDT")
    ap.add_argument("--port", type=int, default=8800, help="port UDP (default 8800)")
    ap.add_argument("--log", metavar="FILE.csv", help="simpan semua paket ke file CSV")
    ap.add_argument("--raw", action="store_true", help="cetak paket mentah, tanpa tabel")
    args = ap.parse_args()

    if os.name == "nt":
        os.system("")       # mengaktifkan kode ANSI di terminal Windows

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.bind(("", args.port))
    sock.settimeout(0.05)

    log_file = writer = None
    if args.log:
        log_file = open(args.log, "w", newline="", encoding="utf-8")
        writer = csv.writer(log_file)
        writer.writerow(["t_unix", "block", "ip", "type", "count", "motion",
                         "euler_x", "euler_y", "euler_z", "a0", "a1", "a2",
                         "face", "state", "stable"])

    blocks = {}
    ip_to_id = {}
    events = deque(maxlen=12)
    last_render = 0.0

    try:
        while True:
            try:
                data, (ip, _) = sock.recvfrom(1024)
            except socket.timeout:
                data = None
            now = time.time()

            if data:
                text = data.decode("ascii", "ignore")
                if args.raw:
                    print(f"{time.strftime('%H:%M:%S')} {ip}  {text.strip()!r}")
                msg = parse(text)
                if msg and msg["type"] == "KBACK":
                    ip_to_id[ip] = msg["id"]
                    blk = blocks.get(f"B{msg['id']}")
                    if blk:
                        blk.ignore_until = now + 1.0     # kalibrasi ulang bukan rotasi
                    events.append(f"{time.strftime('%H:%M:%S')}  B{msg['id']}: kalibrasi "
                                  f"{CAL_RESULT.get(msg['result'], msg['result'])} "
                                  f"(sisi {face_name(msg['face'])})")
                elif msg:
                    # --- tentukan blok ---
                    if msg["id"] is not None:
                        ip_to_id[ip] = msg["id"]
                    bid = msg["id"] if msg["id"] is not None else ip_to_id.get(ip)
                    key = f"B{bid}" if bid is not None else f"IP {ip}"
                    ip_key = f"IP {ip}"
                    if bid is not None and ip_key in blocks and ip_key != key:
                        old = blocks.pop(ip_key)    # entri lama yang hanya dikenal lewat IP
                        if key not in blocks:       # pindahkan datanya ke entri ber-ID
                            old.key = key
                            blocks[key] = old
                    b = blocks.setdefault(key, Block(key))
                    b.ip = ip
                    b.last_seen = now
                    b.rx += 1

                    # --- deteksi paket hilang / blok restart ---
                    c, typ = msg["count"], msg["type"]
                    lc = b.last_count.get(typ)
                    if lc is not None:
                        if c > lc:
                            b.lost += c - lc - 1
                        elif c < lc and now - b.last_restart > 2.0:
                            b.restarts += 1          # KBS4 dan KBS5 sama-sama reset, hitung sekali
                            b.last_restart = now
                            b.state = None
                            events.append(f"{time.strftime('%H:%M:%S')}  {key}: blok restart, "
                                          f"kalibrasi diagonal hilang (kirim 'c' lagi)")
                    b.last_count[typ] = c

                    # --- simpan data terbaru ---
                    if typ == "KBS4":
                        b.motion = msg["motion"]
                        b.euler = msg["euler"]
                        b.analog = msg["analog"]
                    else:
                        b.stable = msg["stable"]
                        if msg["face"] >= 0:
                            new = (msg["face"], msg["code"])
                            if b.state != new:
                                if b.state is not None:
                                    events.append(
                                        f"{time.strftime('%H:%M:%S')}  {key}: "
                                        f"state {b.state[1]} ({face_name(b.state[0])}) -> "
                                        f"{new[1]} ({face_name(new[0])})")
                                    # -- hitung flip / rotasi --#
                                    if now >= b.ignore_until:
                                        old_face, old_code = b.state
                                        new_face, new_code = new
                                        if new_face != old_face:
                                            b.flips += 1
                                            events.append(f"{time.strftime('%H:%M:%S')}  {key}: FLIP " f"{face_name(old_face)} -> {face_name(new_face)}")
                                        elif old_code in DIAG_ORDER and new_code in DIAG_ORDER and old_code != new_code:
                                            step = (DIAG_ORDER.index(new_code) - DIAG_ORDER.index(old_code)) % 4
                                            deg = {1: 90, 2: 180, 3: -90}[step]
                                            b.rots += abs(deg) // 90
                                            b.rot_deg += abs(deg)
                                b.state = new

                    # --- log CSV ---
                    if writer:
                        e = msg.get("euler", ("", "", ""))
                        a = msg.get("analog", ("", "", ""))
                        writer.writerow([f"{now:.3f}", key, ip, typ, c, msg.get("motion", ""),
                                         e[0], e[1], e[2], a[0], a[1], a[2],
                                         msg.get("face", ""), msg.get("code", ""),
                                         msg.get("stable", "")])
                        log_file.flush()

            k = read_key()
            if k:
                sent = None
                if k in ("c", "C"):
                    sent = send_cal(sock, blocks)
                elif k in ("r", "R"):                       # <-- tambahan: reset hitungan (trial baru)
                    for b in blocks.values():
                        b.flips = b.rots = b.rot_deg = 0
                    events.append(f"{time.strftime('%H:%M:%S')}  penghitung flip/rotasi direset (trial baru)")
                elif k in "123456789":
                    sent = send_cal(sock, blocks, int(k))
                if sent is not None:
                    for kk in sent:                         # <-- opsional: abaikan state 2 detik
                        blocks[kk].ignore_until = now + 2.0
                    events.append(f"{time.strftime('%H:%M:%S')}  perintah kalibrasi dikirim ke: " f"{', '.join(sent) if sent else '(tidak ada blok ber-ID)'}")

            if not args.raw and now - last_render >= RENDER_PERIOD_S:
                render(blocks, events, now, args.port, args.log)
                last_render = now

    except KeyboardInterrupt:
        print("\nDihentikan.")
    finally:
        sock.close()
        if log_file:
            log_file.close()


if __name__ == "__main__":
    main()
