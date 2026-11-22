"""Sim Studio: server cho giao diện React (web/joint-lab) — đặt vật/bát, chạy policy trong MuJoCo, lưu lịch sử chạy.

    server.py   HTTP server, IK tĩnh, vùng với tới (lệnh: python -m sim_engine.studio hoặc python -m sim_engine.joint_lab)
    api.py      các endpoint /studio/* (catalog, lịch sử, xem replay đầy đủ)
    runner.py   chạy đúng một ván "lab" từ cảnh đã chỉnh, ghi results/studio/replay_lab.*
    store.py    lưu trữ lịch sử chạy bất biến trong results/studio/studio_runs/
    scene.py    dựng MJCF nhiều vật (hộp/bóng) cho một cảnh Lab
"""
