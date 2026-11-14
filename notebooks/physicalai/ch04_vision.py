import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os
    import sys
    from pathlib import Path

    _here = str(Path(__file__).resolve().parent)
    if _here not in sys.path:
        sys.path.insert(0, _here)

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import physicalai_common as pc
    import cv2
    import mujoco

    np.set_printoptions(precision=3, suppress=True)
    RUN_DIR = pc.make_run_dir("ch04-vision")
    return RUN_DIR, cv2, json, mo, mujoco, np, os, pc, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 4장. 카메라로 물체 찾기: Vision과 Robot 연결

    3장의 Pick and Place는 블록 위치를 시뮬레이터에서 **정답 그대로** 읽었습니다. 실제 로봇은 그럴 수 없습니다.
    이 장에서는 카메라 이미지로 블록 위치를 추정하고, 그 추정값으로 같은 Pick and Place를 실행합니다.

    ```mermaid
    flowchart LR
      Cam[Camera<br/>RGB + Depth] --> Det[Color Detection<br/>픽셀 u, v]
      Det --> Pos[Pixel → World<br/>x, y, z]
      Pos --> Ctrl[3장 Pick and Place]
      Ctrl --> Arm[Robot Arm]
    ```

    | 절 | 내용 |
    | --- | --- |
    | 4-1 | OpenCV 이미지 기초: 배열, BGR/RGB/HSV |
    | 4-2 | HSV 색 검출과 영상 속 추적 |
    | 4-3 | MuJoCo 가상 카메라: RGB·Depth, 핀홀 모델, 픽셀 ↔ 월드 |
    | 4-4 | 카메라로 블록 위치 추정과 오차 |
    | 4-5 | (선택) YOLO 사전학습 검출기 |
    | 4-6 | Vision Pick and Place와 인식 오차 실험 |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4-1. OpenCV로 이미지 다루기

    실물 카메라 대신 시뮬레이터의 `fixed` 카메라로 찍은 장면을 씁니다. 이미지는 `(높이, 너비, 채널)` 모양의 `uint8` 배열입니다.
    **주의**: MuJoCo·matplotlib·imageio는 RGB 순서, OpenCV의 `imread`/`imwrite`/`VideoCapture`는 **BGR** 순서입니다.
    """)
    return


@app.cell
def _(cv2, mo, np, pc):
    _robot = pc.SO101Sim(cube_pos=(0.24, 0.0), box_pos=pc.BOX_XY, render=False)
    scene_rgb = _robot.snapshot(camera="fixed")
    _robot.close()
    _bgr = cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2BGR)
    _gray = cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2GRAY)
    _hsv = cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2HSV)
    _small = cv2.resize(scene_rgb, (scene_rgb.shape[1] // 4, scene_rgb.shape[0] // 4))
    _v, _u = 240, 320
    mo.vstack([
        mo.md(f"""
    - shape `{scene_rgb.shape}`, dtype `{scene_rgb.dtype}`, 값 범위 {scene_rgb.min()}~{scene_rgb.max()}
    - 픽셀 (v={_v}, u={_u}): RGB `{scene_rgb[_v, _u]}`, BGR `{_bgr[_v, _u]}`, GRAY `{_gray[_v, _u]}`, HSV `{_hsv[_v, _u]}`
    - `cv2.resize`로 1/4 축소: `{_small.shape}`. 인덱스는 `[세로 v, 가로 u]` 순서입니다.
    """),
        mo.hstack([mo.image(scene_rgb, width=300, caption="RGB"),
                   mo.image(_bgr, width=300, caption="BGR를 RGB로 착각하고 표시"),
                   mo.image(_gray, width=300, caption="GRAY"),
                   mo.image(_hsv[:, :, 0], width=300, caption="HSV의 H(색상) 채널")], wrap=True),
    ])
    return (scene_rgb,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    두 번째 그림은 빨간 블록이 파랗게 보입니다. 채널 순서를 바꾸지 않고 표시하면 생기는 흔한 실수입니다.

    ## 4-2. HSV 색 공간에서 빨간색 찾기

    RGB는 조명에 따라 세 값이 함께 변해 "빨강"의 범위를 정하기 어렵습니다. **HSV**는 색상(H)·채도(S)·밝기(V)로 나눠
    H만 보면 색을 고를 수 있습니다. OpenCV의 H는 0~179이고 **빨강은 0 근처와 179 근처 두 구간**에 걸쳐 있습니다.

    처리 순서: HSV 변환 → `inRange`로 마스크 → `morphologyEx(OPEN)`으로 작은 점 제거 → 가장 큰 윤곽(contour) → 모멘트로 중심.
    슬라이더로 범위를 바꿔 마스크가 어떻게 변하는지 보세요.
    """)
    return


@app.cell
def _(mo):
    hsv_h1 = mo.ui.range_slider(0, 179, value=[0, 10], label="H 구간 1")
    hsv_h2 = mo.ui.range_slider(0, 179, value=[170, 179], label="H 구간 2")
    hsv_s = mo.ui.slider(0, 255, value=120, label="S 최솟값")
    hsv_v = mo.ui.slider(0, 255, value=70, label="V 최솟값")
    mo.vstack([mo.hstack([hsv_h1, hsv_h2], justify="start"), mo.hstack([hsv_s, hsv_v], justify="start")])
    return hsv_h1, hsv_h2, hsv_s, hsv_v


@app.cell
def _(cv2, hsv_h1, hsv_h2, hsv_s, hsv_v, mo, np, scene_rgb):
    def detect_hsv(rgb, ranges, min_area=30):
        """HSV 범위 목록 → (검출 dict 또는 None, 마스크)."""
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, mask
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < min_area:
            return None, mask
        m = cv2.moments(c)
        return {"center": (m["m10"] / m["m00"], m["m01"] / m["m00"]), "area": cv2.contourArea(c),
                "bbox": cv2.boundingRect(c)}, mask

    hsv_ranges = [((hsv_h1.value[0], hsv_s.value, hsv_v.value), (hsv_h1.value[1], 255, 255)),
                  ((hsv_h2.value[0], hsv_s.value, hsv_v.value), (hsv_h2.value[1], 255, 255))]
    _det, _mask = detect_hsv(scene_rgb, hsv_ranges)
    _marked = scene_rgb.copy()
    if _det is not None:
        _x, _y, _w, _h = _det["bbox"]
        cv2.rectangle(_marked, (_x, _y), (_x + _w, _y + _h), (0, 255, 0), 2)
        cv2.circle(_marked, tuple(int(round(c)) for c in _det["center"]), 4, (255, 255, 0), -1)
    _info = (f"중심 (u, v) = ({_det['center'][0]:.1f}, {_det['center'][1]:.1f}), 면적 {_det['area']:.0f} px, "
             f"마스크 픽셀 {int((_mask > 0).sum())}") if _det else "검출 없음"
    mo.vstack([mo.hstack([mo.image(_marked, width=380, caption="검출 결과"),
                          mo.image(_mask, width=380, caption="마스크")]), mo.md(_info)])
    return detect_hsv, hsv_ranges


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    S 최솟값을 0으로 내리면 회색 바닥까지 잡히고, 구간 2를 끄면(170~179 → 179~179) 빨강의 일부를 놓칠 수 있습니다.

    ### 영상 속 블록 추적

    3장 Pick and Place를 `fixed` 카메라로 찍으며 매 프레임 빨간 블록을 찾습니다. 그리퍼가 블록을 가리는 순간
    면적이 줄거나 검출이 끊기는지 확인하세요.
    """)
    return


@app.cell
def _(RUN_DIR, cv2, detect_hsv, hsv_ranges, mo, np, pc, plt):
    _robot = pc.SO101Sim(cube_pos=(0.24, 0.0), box_pos=pc.BOX_XY, render=True, camera="fixed")
    pc.scripted_pick_and_place(_robot)
    _track, _out = [], []
    for _f in _robot.frames:
        _d, _ = detect_hsv(_f, hsv_ranges)
        _img = _f.copy()
        if _d:
            _x, _y, _w, _h = _d["bbox"]
            cv2.rectangle(_img, (_x, _y), (_x + _w, _y + _h), (0, 255, 0), 2)
            _track.append((*_d["center"], _d["area"]))
        else:
            _track.append((np.nan, np.nan, 0.0))
        _out.append(_img)
    _robot.close()
    _track = np.array(_track)
    _video = pc.save_video(_out, RUN_DIR / "color_track.mp4", fps=30)
    _t = np.arange(len(_track)) / 30
    _fig, _axes = plt.subplots(2, 1, figsize=(5.5, 3.6), sharex=True)
    _axes[0].plot(_t, _track[:, 0], label="u")
    _axes[0].plot(_t, _track[:, 1], label="v")
    _axes[0].set(ylabel="pixel")
    _axes[0].legend(fontsize=8)
    _axes[1].plot(_t, _track[:, 2], color="tab:red")
    _axes[1].set(xlabel="time (s)", ylabel="area (px)")
    for _ax in _axes:
        _ax.grid(alpha=0.3)
    _fig.tight_layout()
    mo.vstack([mo.hstack([mo.video(src=_video, loop=True, width=420), _fig], align="center"),
               mo.md(f"검출 {int(np.isfinite(_track[:, 0]).sum())}/{len(_track)} 프레임")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4-3. 가상 카메라: RGB와 Depth

    시뮬레이터는 카메라마다 RGB와 **Depth**(카메라 축 방향 거리, m)를 함께 줍니다. 실제로는 RealSense 같은 RGB-D 카메라가
    같은 역할을 합니다. 장면에는 카메라 6대가 있습니다. `wrist_cam`은 SO-101 모델에 달린 손목 카메라라서 팔과 함께 움직입니다.
    """)
    return


@app.cell
def _(cv2, mo, mujoco, np, pc):
    def depth_to_color(depth, max_m=1.0):
        """가까울수록 빨갛게 보이는 Depth 시각화."""
        d = np.clip(depth / max_m, 0, 1)
        return cv2.cvtColor(cv2.applyColorMap((255 * (1 - d)).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)

    _robot = pc.SO101Sim(cube_pos=(0.24, 0.0), box_pos=pc.BOX_XY, render=False)
    _m, _d = _robot.model, _robot.data
    _rows = ["| id | 카메라 | 위치 (m) | fovy |", "| ---: | --- | --- | ---: |"]
    for _c in range(_m.ncam):
        _rows.append(f"| {_c} | `{mujoco.mj_id2name(_m, mujoco.mjtObj.mjOBJ_CAMERA, _c)}` | "
                     f"{np.round(_d.cam_xpos[_c], 3)} | {_m.cam_fovy[_c]:.0f}° |")
    _imgs = []
    for _name in ["top", "fixed", "side", "wrist_cam"]:
        _cam = pc.SimCamera(_m, _name, 320, 240)
        _rgb, _depth = _cam.capture(_d)
        _imgs += [mo.image(_rgb, caption=f"{_name} RGB"),
                  mo.image(depth_to_color(_depth), caption=f"{_name} depth {_depth.min():.2f}~{_depth.max():.2f} m")]
        _cam.close()
    # 손목 카메라는 대기 자세에서 하늘을 본다. 블록 위로 손을 보낸 뒤 다시 찍는다
    _robot.open_gripper(0.5)
    _robot.move_to([0.24, 0.0, 0.08], seconds=1.2)
    _cam = pc.SimCamera(_m, "wrist_cam", 320, 240)
    _rgb, _depth = _cam.capture(_d)
    _cam.close()
    _robot.close()
    _imgs += [mo.image(_rgb, caption="wrist_cam: 블록 위 8 cm"),
              mo.image(depth_to_color(_depth, 0.3), caption="wrist_cam depth (0~0.3 m)")]
    mo.vstack([mo.md("\n".join(_rows)), mo.hstack(_imgs, wrap=True, widths="equal")])
    return (depth_to_color,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 핀홀 카메라 모델: 픽셀 ↔ 월드

    세로 시야각 `fovy`와 이미지 높이 $H$로 초점 거리(픽셀 단위)를 구합니다: $f = \dfrac{H/2}{\tan(\text{fovy}/2)}$.
    픽셀 $(u, v)$와 그 픽셀의 depth $d$가 있으면 카메라 좌표계의 점은

    $$
    x_c = \frac{u - c_x}{f}d,\quad y_c = -\frac{v - c_y}{f}d,\quad z_c = -d
    $$

    입니다(MuJoCo 카메라는 $-z$ 방향을 보고 $y$가 위, 이미지 $v$는 아래가 +). 월드 좌표는 카메라 회전 $R$과 위치 $t$로
    $p_w = R\,p_c + t$. 반대로 월드 점을 이미지에 투영하면 $u = c_x + f\,x_c/(-z_c)$, $v = c_y - f\,y_c/(-z_c)$입니다.
    """)
    return


@app.cell
def _(cv2, mo, np, pc):
    _robot = pc.SO101Sim(cube_pos=(0.24, 0.0), box_pos=pc.BOX_XY, render=False)
    _cam = pc.SimCamera(_robot.model, "top", 640, 480)
    _d = _robot.data
    _rgb, _depth = _cam.capture(_d)
    _R = _d.cam_xmat[_cam.cam_id].reshape(3, 3)
    _t = _d.cam_xpos[_cam.cam_id]
    # 블록 윗면 중심을 이미지에 투영했다가 depth로 되돌린다
    _cube_top = _robot.cube_pos() + [0, 0, 0.015]
    _pc = _R.T @ (_cube_top - _t)
    _u, _v = _cam.cx + _cam.f * _pc[0] / (-_pc[2]), _cam.cy - _cam.f * _pc[1] / (-_pc[2])
    _back = _cam.pixel_to_world(_d, _u, _v, _depth[int(round(_v)), int(round(_u))])
    _center = _cam.pixel_to_world(_d, 320, 240, _depth[240, 320])
    _marked = _rgb.copy()
    cv2.circle(_marked, (int(round(_u)), int(round(_v))), 8, (0, 255, 0), 2)
    _cam.close()
    _robot.close()
    mo.hstack([mo.image(_marked, width=380), mo.md(f"""
    - 초점 거리 f = {_cam.f:.1f} px (fovy {_robot.model.cam_fovy[_cam.cam_id]:.0f}°, 세로 480 px)
    - 카메라 위치 {np.round(_t, 3)}, 회전행렬의 z축 {np.round(_R[:, 2], 3)} (카메라는 −z를 본다 = 아래)
    - 중앙 픽셀 (320, 240) depth {_depth[240, 320]:.3f} m → 월드 {np.round(_center, 3)}
    - 블록 윗면 중심 {np.round(_cube_top, 3)} → 픽셀 ({_u:.1f}, {_v:.1f})
    - 그 픽셀의 depth로 되돌린 월드 좌표 {np.round(_back, 4)}, 오차 **{np.linalg.norm(_back - _cube_top) * 1000:.1f} mm**
    """)], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4-4. 카메라로 블록 위치 추정하기

    색 검출(중심 픽셀) + depth(중심 주변 3×3의 중앙값) + 역투영을 합친 것이 `physicalai_common.locate_object`입니다.
    블록을 무작위 위치에 놓고 추정값과 정답을 비교합니다. 추정한 z는 블록 **윗면**(0.03 m) 근처가 나옵니다.
    위에서 내려다보면 윗면만 보이기 때문입니다. 집을 때는 x, y만 쓰고 높이는 상수(`GRASP_Z`)를 씁니다.
    """)
    return


@app.cell
def _(mo):
    loc_camera = mo.ui.dropdown(["top", "top_zoom", "fixed"], value="top", label="카메라")
    loc_res = mo.ui.dropdown({"640x480": (640, 480), "320x240": (320, 240), "160x120": (160, 120)},
                             value="640x480", label="해상도")
    mo.hstack([loc_camera, loc_res], justify="start")
    return loc_camera, loc_res


@app.cell
def _(loc_camera, loc_res, mo, np, pc):
    _rng = np.random.default_rng(0)
    _rows = ["| 정답 (x, y) | 픽셀 (u, v) | 추정 (x, y, z) | xy 오차 (mm) |", "| --- | --- | --- | ---: |"]
    _errs = []
    for _i in range(8):
        _xy = (0.24, 0.0) if _i == 0 else pc.random_cube(_rng)
        _robot = pc.SO101Sim(cube_pos=_xy, box_pos=pc.BOX_XY, render=False)
        _cam = pc.SimCamera(_robot.model, loc_camera.value, *loc_res.value)
        _est, _det = pc.locate_object(_cam, _robot.data, "red")
        _true = _robot.cube_pos()
        if _est is None:
            _rows.append(f"| {np.round(_true[:2], 3)} | 검출 실패 | | |")
        else:
            _e = np.linalg.norm(_est[:2] - _true[:2]) * 1000
            _errs.append(_e)
            _rows.append(f"| {np.round(_true[:2], 3)} | ({_det['center'][0]:.1f}, {_det['center'][1]:.1f}) | "
                         f"{np.round(_est, 3)} | {_e:.1f} |")
        _cam.close()
        _robot.close()
    _errs = np.array(_errs)
    _summary = (f"검출 {len(_errs)}/8, xy 오차 평균 **{_errs.mean():.1f} mm**, 최대 {_errs.max():.1f} mm"
                if len(_errs) else "검출 0/8")
    mo.md("\n".join(_rows) + "\n\n" + _summary +
          "\n\n`fixed`처럼 비스듬한 카메라는 블록 옆면까지 보여 중심 픽셀이 실제 중심과 어긋납니다. 해상도를 낮추면 오차가 커집니다.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4-5. (선택) YOLO 사전학습 검출기

    색 검출은 "빨간 것"만 찾을 수 있습니다. **YOLO**는 학습된 신경망으로 여러 종류의 물체를 한 번에 찾습니다.
    이 절은 `ultralytics`가 필요합니다: `uv sync --locked --extra vision`.
    처음 실행하면 `yolo11n.pt`(약 5 MB)를 `outputs/physicalai/cache/yolo/`에 내려받습니다.

    COCO 80개 클래스로 학습된 모델이라 사람·버스는 잘 찾지만, 시뮬레이터의 "빨간 정육면체"는 클래스에 없어 대부분 놓치거나
    엉뚱한 이름을 붙입니다. 학습 데이터에 없는 물체는 찾지 못한다는 것이 이 실험의 요점입니다.
    원본 저장소의 `ch04/yolo_pick_and_place.py`는 참조 이미지에 박스를 한 번 그려 주는 YOLOE visual prompt로 이 문제를 풉니다.
    """)
    return


@app.cell
def _(mo):
    yolo_btn = mo.ui.run_button(label="YOLO 실행 (yolo11n.pt 다운로드)")
    yolo_btn
    return (yolo_btn,)


@app.cell
def _(RUN_DIR, cv2, mo, os, pc, yolo_btn):
    mo.stop(not yolo_btn.value, mo.md("버튼을 누르면 실행합니다."))
    _cache = pc.OUTPUT_ROOT / "cache" / "yolo"
    _cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(_cache))
    try:
        from ultralytics import YOLO, settings
    except ImportError:
        mo.stop(True, mo.md("`ultralytics`가 없습니다. `uv sync --locked --extra vision` 후 커널을 다시 시작하세요."))
    settings.update({"weights_dir": str(_cache), "sync": False})
    _model = YOLO(str(_cache / "yolo11n.pt"))
    _outs = []
    _robot = pc.SO101Sim(cube_pos=(0.24, 0.0), box_pos=pc.BOX_XY, render=False)
    _bus = _cache / "bus.jpg"
    if not _bus.is_file():
        import urllib.request

        urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", _bus)
    _sources = [("bus", "실제 사진", str(_bus)),
                ("sim_fixed", "시뮬레이터 fixed 카메라", cv2.cvtColor(_robot.snapshot(camera="fixed"), cv2.COLOR_RGB2BGR)),
                ("sim_top", "시뮬레이터 top 카메라", cv2.cvtColor(_robot.snapshot(camera="top"), cv2.COLOR_RGB2BGR))]
    _robot.close()
    for _tag, _title, _src in _sources:
        _r = _model.predict(_src, conf=0.25, verbose=False)[0]   # ndarray 입력은 BGR 순서
        _names = [f"{_model.names[int(b.cls)]} {float(b.conf):.2f}" for b in _r.boxes]
        cv2.imwrite(str(RUN_DIR / f"yolo_{_tag}.png"), _r.plot())
        _outs.append(mo.vstack([mo.image(cv2.cvtColor(_r.plot(), cv2.COLOR_BGR2RGB), width=320),
                                mo.md(f"**{_title}**: {', '.join(_names) or '검출 없음'}")]))
    mo.hstack(_outs, wrap=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4-6. Vision Pick and Place

    3장 `pick_and_place`에서 바뀌는 것은 **블록·상자 위치를 어디서 얻는가** 한 곳뿐입니다.

    ```python
    cube = robot.cube_pos()                       # 3장: 시뮬레이터의 정답
    cube, _ = locate_object(cam, data, "red")     # 4장: 카메라 추정
    ```

    아래에서는 추정값에 일부러 가우시안 잡음을 더해, 인식 오차가 몇 mm부터 집기에 실패하는지도 확인합니다.
    """)
    return


@app.cell
def _(mo):
    vpp_noise = mo.ui.slider(0, 20, value=0, step=1, label="추정 위치에 더할 잡음 표준편차 (mm)")
    vpp_trials = mo.ui.slider(5, 30, value=10, step=5, label="시도 횟수")
    vpp_btn = mo.ui.run_button(label="Vision Pick and Place 반복")
    mo.hstack([vpp_noise, vpp_trials, vpp_btn], justify="start")
    return vpp_btn, vpp_noise, vpp_trials


@app.cell
def _(np, pc):
    def perceive(robot, cam, noise_m=0.0, rng=None):
        """카메라 한 장으로 블록과 상자의 위치를 추정한다 (Perception)."""
        rgb, depth = cam.capture(robot.data)
        cube, _ = pc.locate_object(cam, robot.data, "red", rgb, depth)
        box, _ = pc.locate_object(cam, robot.data, "blue", rgb, depth)
        if cube is not None and noise_m > 0:
            cube = cube + np.r_[rng.normal(0, noise_m, 2), 0.0]
        return cube, box


    def vision_pick_and_place(robot, cam, noise_m=0.0, rng=None):
        cube, box = perceive(robot, cam, noise_m, rng)
        if cube is None or box is None:
            return False, None
        err = float(np.linalg.norm(cube[:2] - robot.cube_pos()[:2]))   # 평가 기록에만 쓰는 정답
        return pc.scripted_pick_and_place(robot, target=cube, box=box), err
    return (vision_pick_and_place,)


@app.cell
def _(RUN_DIR, mo, pc, vision_pick_and_place):
    _robot = pc.SO101Sim(cube_pos=(0.21, -0.1), box_pos=pc.BOX_XY, render=True, camera="fixed")
    _cam = pc.SimCamera(_robot.model, "top")
    _ok, _err = vision_pick_and_place(_robot, _cam)
    _cam.close()
    _video = _robot.save_video(RUN_DIR / "vision_pick_and_place.mp4")
    _robot.close()
    mo.hstack([mo.video(src=_video, loop=True, width=440),
               mo.md(f"블록 (0.21, −0.10), 카메라 추정 오차 {_err * 1000:.1f} mm → **{'성공' if _ok else '실패'}**")],
              align="center")
    return


@app.cell
def _(
    RUN_DIR,
    json,
    mo,
    np,
    pc,
    vision_pick_and_place,
    vpp_btn,
    vpp_noise,
    vpp_trials,
):
    mo.stop(not vpp_btn.value, mo.md("버튼을 누르면 무작위 위치에서 반복합니다(10회 약 10초)."))
    _rng = np.random.default_rng(0)
    _noise_rng = np.random.default_rng(1)
    _records = []
    with mo.status.progress_bar(total=vpp_trials.value, title="Vision Pick and Place") as _bar:
        for _i in range(vpp_trials.value):
            _cube = pc.random_cube(_rng)
            _robot = pc.SO101Sim(cube_pos=_cube, box_pos=pc.BOX_XY, render=False)
            _cam = pc.SimCamera(_robot.model, "top")
            _ok, _err = vision_pick_and_place(_robot, _cam, vpp_noise.value / 1000, _noise_rng)
            _records.append({"cube_xy": list(_cube), "success": bool(_ok), "xy_error_m": _err})
            _cam.close()
            _robot.close()
            _bar.update()
    (RUN_DIR / f"vision_pick_noise{vpp_noise.value}mm.json").write_text(json.dumps(_records, indent=2))
    _ok = sum(r["success"] for r in _records)
    _errs = [r["xy_error_m"] * 1000 for r in _records if r["xy_error_m"] is not None]
    mo.md(f"### 잡음 {vpp_noise.value} mm: 성공률 {_ok}/{len(_records)} = {100 * _ok / len(_records):.0f}%\n\n"
          f"실제 위치 오차 평균 {np.mean(_errs):.1f} mm, 최대 {np.max(_errs):.1f} mm. "
          "잡음을 5, 10, 15 mm로 올려 가며 성공률이 떨어지는 지점을 찾아보세요. 그리퍼 손가락 사이 여유가 한계입니다.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 정리와 다음 장

    - Vision 파이프라인: 이미지 → 색 마스크 → 중심 픽셀 → depth → 핀홀 역투영 → 월드 좌표.
    - 색 검출은 빠르고 정확하지만 조명·바닥 색·비슷한 색 물체에 약합니다(8장에서 다시 확인합니다).
    - 인식 오차는 곧바로 집기 실패로 이어집니다. 인식과 제어를 따로 설계한 파이프라인은 각 단계의 오차가 쌓입니다.

    **확인 문제**

    1. `COLOR_RANGES["red"]`에서 두 번째 구간(170~179)을 빼면 검출 결과가 어떻게 바뀌나요?
    2. depth 대신 "블록 윗면은 항상 z = 0.03"이라는 가정으로 월드 좌표를 구하려면 어떻게 계산해야 할까요?

    다음 [5장](ch05_reinforcement_learning.py)에서는 사람이 규칙을 쓰는 대신 보상만 주고 행동을 학습시킵니다.
    """)
    return


if __name__ == "__main__":
    app.run()
