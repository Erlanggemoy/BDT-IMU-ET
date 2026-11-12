"""
eyetracker_module.py  (Stream Engine edition)
==============================================
Modul integrasi Tobii Eye Tracker 5 untuk Block Design Test (BDT-2025),
menggunakan Tobii Stream Engine (low-level, tanpa batasan lisensi
"analytical" yang memblokir tobii_research pada consumer device).

PRASYARAT:
    - File `tobii_stream_engine.dll` (64-bit) harus ada satu folder dengan
      file ini, atau di folder `DLL/` relatif terhadap file ini, atau ada
      di PATH sistem.

Cara pakai singkat (SAMA seperti versi tobii_research sebelumnya):
    from eyetracker_module import EyeTrackerHandler

    et = EyeTrackerHandler(on_gaze=callback_fn)
    et.connect()          # cari & konek ke eye tracker
    et.start()             # mulai subscribe gaze data (tombol "Start")
    et.stop()               # berhenti subscribe (tombol "Stop")

`callback_fn(gaze_point)` dipanggil setiap ada data gaze baru, dengan
gaze_point berupa dict: {"x": float (0.0-1.0), "y": float (0.0-1.0),
"timestamp": int (microseconds)}.
"""

import ctypes
import os
import threading
import time


# ---------------------------------------------------------------------
# Cari & load tobii_stream_engine.dll
# ---------------------------------------------------------------------
def _find_and_load_dll():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "tobii_stream_engine.dll"),
        os.path.join(here, "DLL", "tobii_stream_engine.dll"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ctypes.CDLL(path)
            except OSError as e:
                raise OSError(f"Gagal load DLL di {path}: {e}") from e

    # Terakhir, coba biarkan Windows cari lewat PATH
    try:
        return ctypes.CDLL("tobii_stream_engine.dll")
    except OSError as e:
        raise OSError(
            "tobii_stream_engine.dll tidak ditemukan. Taruh file itu di "
            f"folder yang sama dengan {__file__}, atau di subfolder DLL/, "
            "atau tambahkan ke PATH sistem."
        ) from e


_lib = _find_and_load_dll()


# ---------------------------------------------------------------------
# Struct & typedef sesuai tobii.h / tobii_streams.h
# ---------------------------------------------------------------------
class _GazePoint(ctypes.Structure):
    _fields_ = [
        ("timestamp_us", ctypes.c_int64),
        ("validity", ctypes.c_int),       # tobii_validity_t (enum -> int)
        ("position_xy", ctypes.c_float * 2),
    ]


_GAZE_POINT_CALLBACK = ctypes.CFUNCTYPE(
    None, ctypes.POINTER(_GazePoint), ctypes.c_void_p
)
_URL_RECEIVER_CALLBACK = ctypes.CFUNCTYPE(
    None, ctypes.c_char_p, ctypes.c_void_p
)

TOBII_ERROR_NO_ERROR = 0
TOBII_VALIDITY_VALID = 1

# ---------------------------------------------------------------------
# Signature fungsi C (biar ctypes tidak salah tebak tipe & crash)
# ---------------------------------------------------------------------
_lib.tobii_api_create.argtypes = [
    ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p
]
_lib.tobii_api_create.restype = ctypes.c_int

_lib.tobii_api_destroy.argtypes = [ctypes.c_void_p]
_lib.tobii_api_destroy.restype = ctypes.c_int

_lib.tobii_enumerate_local_device_urls.argtypes = [
    ctypes.c_void_p, _URL_RECEIVER_CALLBACK, ctypes.c_void_p
]
_lib.tobii_enumerate_local_device_urls.restype = ctypes.c_int

_lib.tobii_device_create.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)
]
_lib.tobii_device_create.restype = ctypes.c_int

_lib.tobii_device_destroy.argtypes = [ctypes.c_void_p]
_lib.tobii_device_destroy.restype = ctypes.c_int

_lib.tobii_wait_for_callbacks.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)
]
_lib.tobii_wait_for_callbacks.restype = ctypes.c_int

_lib.tobii_device_process_callbacks.argtypes = [ctypes.c_void_p]
_lib.tobii_device_process_callbacks.restype = ctypes.c_int

_lib.tobii_gaze_point_subscribe.argtypes = [
    ctypes.c_void_p, _GAZE_POINT_CALLBACK, ctypes.c_void_p
]
_lib.tobii_gaze_point_subscribe.restype = ctypes.c_int

_lib.tobii_gaze_point_unsubscribe.argtypes = [ctypes.c_void_p]
_lib.tobii_gaze_point_unsubscribe.restype = ctypes.c_int


class EyeTrackerHandler:
    def __init__(self, on_gaze=None, on_status_change=None):
        self.on_gaze = on_gaze
        self.on_status_change = on_status_change

        self._api = ctypes.c_void_p()
        self._device = ctypes.c_void_p()
        self._device_url = None

        self._is_tracking = False
        self._stop_thread_flag = threading.Event()
        self._process_thread = None
        self._lock = threading.Lock()

        # Simpan referensi callback ctypes supaya tidak di-garbage-collect
        self._gaze_callback_ref = _GAZE_POINT_CALLBACK(self._gaze_callback)
        self._url_receiver_ref = _URL_RECEIVER_CALLBACK(self._url_receiver)

        self._found_urls = []
        self._gaze_duration_us = 0
        self._last_gaze_ts = None

    # ------------------------------------------------------------------
    def connect(self, timeout=5.0):
        """Inisialisasi API + cari device. Return True kalau berhasil."""
        self._notify_status("searching")

        err = _lib.tobii_api_create(ctypes.byref(self._api), None, None)
        if err != TOBII_ERROR_NO_ERROR:
            self._notify_status(f"error: tobii_api_create gagal (code {err})")
            return False

        start = time.time()
        while time.time() - start < timeout:
            self._found_urls = []
            _lib.tobii_enumerate_local_device_urls(
                self._api, self._url_receiver_ref, None
            )
            if self._found_urls:
                break
            time.sleep(0.5)

        if not self._found_urls:
            self._notify_status("error: eye tracker tidak ditemukan")
            return False

        self._device_url = self._found_urls[0]
        err = _lib.tobii_device_create(
            self._api, self._device_url, ctypes.byref(self._device)
        )
        if err != TOBII_ERROR_NO_ERROR:
            self._notify_status(f"error: tobii_device_create gagal (code {err})")
            return False

        self._notify_status(f"connected: {self._device_url.decode('utf-8')}")
        return True

    def _url_receiver(self, url, user_data):
        if url:
            self._found_urls.append(url)

    # ------------------------------------------------------------------
    def start(self):
        """Mulai subscribe gaze data + thread pemroses callback."""
        if not self._device:
            self._notify_status("error: belum connect ke eye tracker")
            return False

        with self._lock:
            if self._is_tracking:
                return True

            self._gaze_duration_us = 0
            self._last_gaze_ts = None

            err = _lib.tobii_gaze_point_subscribe(
                self._device, self._gaze_callback_ref, None
            )
            if err != TOBII_ERROR_NO_ERROR:
                self._notify_status(f"error: subscribe gagal (code {err})")
                return False

            self._is_tracking = True
            self._stop_thread_flag.clear()
            self._process_thread = threading.Thread(
                target=self._process_loop, daemon=True
            )
            self._process_thread.start()

        self._notify_status("tracking")
        return True

    def stop(self):
        """Berhenti subscribe. Tidak menunggu thread selesai (non-blocking),
        supaya tidak membekukan GUI. Thread background akan berhenti sendiri
        begitu melihat flag ini (dalam <1 detik)."""
        with self._lock:
            if not self._is_tracking:
                return
            self._stop_thread_flag.set()
            _lib.tobii_gaze_point_unsubscribe(self._device)
            self._is_tracking = False

        self._notify_status("stopped")

    def disconnect(self):
        """Bersihkan device & api sepenuhnya (panggil saat aplikasi ditutup)."""
        self.stop()
        # Pastikan thread background benar-benar berhenti sebelum device
        # di-destroy, supaya tidak ada akses ke memori yang sudah dibebaskan.
        if self._process_thread is not None:
            self._process_thread.join(timeout=2.0)
        if self._device:
            _lib.tobii_device_destroy(self._device)
            self._device = ctypes.c_void_p()
        if self._api:
            _lib.tobii_api_destroy(self._api)
            self._api = ctypes.c_void_p()

    @property
    def is_tracking(self):
        return self._is_tracking

    @property
    def gaze_duration_seconds(self):
        return self._gaze_duration_us / 1_000_000

    # ------------------------------------------------------------------
    def _process_loop(self):
        """
        Loop wajib Stream Engine: tobii_device_process_callbacks() harus
        dipanggil rutin (idealnya >=10x/detik) supaya data & koneksi
        tetap hidup. tobii_wait_for_callbacks() dipakai supaya tidak
        busy-wait / makan CPU percuma.
        """
        device_arr = (ctypes.c_void_p * 1)(self._device.value)
        while not self._stop_thread_flag.is_set():
            # engine=None (NULL) karena kita tidak pakai tobii_engine_t
            err = _lib.tobii_wait_for_callbacks(None, 1, device_arr)
            # TOBII_ERROR_TIMED_OUT itu NORMAL (cuma berarti belum ada data
            # baru dalam beberapa ratus ms), bukan kegagalan -- lanjut saja.
            if err == TOBII_ERROR_NO_ERROR:
                _lib.tobii_device_process_callbacks(self._device)

    def _gaze_callback(self, gaze_point_ptr, user_data):
        gp = gaze_point_ptr.contents
        if gp.validity != TOBII_VALIDITY_VALID:
            return

        current_ts = gp.timestamp_us
        if self._last_gaze_ts is not None:
            delta = current_ts - self._last_gaze_ts
            if 0 < delta < 500_000:
                self._gaze_duration_us += delta
        self._last_gaze_ts = current_ts

        result = {
            "x": gp.position_xy[0],
            "y": gp.position_xy[1],
            "timestamp": gp.timestamp_us,
        }
        if self.on_gaze:
            self.on_gaze(result)

    def _notify_status(self, status):
        print(f"[EyeTracker] {status}")
        if self.on_status_change:
            self.on_status_change(status)


# ----------------------------------------------------------------------
# Contoh pemakaian mandiri (jalankan file ini langsung untuk test cepat)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    count = [0]

    def handle_gaze(g):
        count[0] += 1
        if count[0] <= 30:  # biar tidak spam terminal
            px = int(g["x"] * 1920)
            py = int(g["y"] * 1080)
            print(f"Gaze -> screen px: ({px}, {py})  t={g['timestamp']}")

    def handle_status(s):
        print(f"Status berubah: {s}")

    et = EyeTrackerHandler(on_gaze=handle_gaze, on_status_change=handle_status)

    if et.connect():
        et.start()
        try:
            print("Tracking selama 10 detik, tatap layar...")
            time.sleep(10)
        finally:
            et.stop()
            et.disconnect()
            print(f"Total gaze data diterima: {count[0]}")
