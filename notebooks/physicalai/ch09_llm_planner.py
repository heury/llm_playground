import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import base64
    import io
    import json
    import os
    import re
    import sys
    import time
    from pathlib import Path

    _here = str(Path(__file__).resolve().parent)
    if _here not in sys.path:
        sys.path.insert(0, _here)

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import physicalai_common as pc

    RUN_DIR = pc.make_run_dir("ch09-llm")
    return Path, RUN_DIR, base64, io, json, mo, np, os, pc, plt, re, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 9장. LLM이 지휘하는 로봇: LLM Planner

    8장에서 SmolVLA는 "빨강·초록·노랑 중 문장이 가리키는 색"을 고르는 데 어려움을 겪었습니다.
    이번에는 정책을 학습하지 않습니다. **LLM에게 3장의 로봇 기능을 도구(tool)로 쥐여 주고** 무엇을 어떤 순서로 할지만 정하게 합니다.

    ```mermaid
    flowchart TB
      L["LLM (0.2~2 Hz)<br/>어느 블록을, 어떤 순서로"] -->|도구 호출| T["RobotTools<br/>목표점 검사 · 안전 범위 제한"]
      T -->|결과 JSON| L
      T --> I["SO101Sim (500 Hz)<br/>3장 IK · 관절 보간"]
    ```

    1장의 세 줄이 그대로입니다. `policy` 자리에 LLM이 들어갔을 뿐입니다.

    ```python
    obs = get_scene_state()      # 관찰  ← 도구 호출의 결과
    action = LLM(obs)            # 판단  ← 다음에 부를 도구를 LLM이 고른다
    robot.execute(action)        # 행동  ← 3장의 IK와 관절 보간
    ```

    이 장은 **API 키 없이도** 진행할 수 있도록 '두뇌'를 세 가지로 준비했습니다.

    | 두뇌 | 설명 | 필요한 것 |
    | --- | --- | --- |
    | `scripted` | 같은 도구 프로토콜을 쓰는 규칙 기반 Planner. 루프 구조 확인용 | 없음 |
    | `replay` | 원본 책 저자가 실제 LLM(gpt-6-astra)으로 측정한 대화 기록(trace)의 도구 호출을 그대로 재생 | 릴리스 trace 90 KB |
    | `openai` | 실제 LLM API 호출 (Responses API) | `uv sync --locked --extra llm`, `OPENAI_API_KEY`, 요금 |

    성공 여부는 LLM의 주장(`report_done`)이 아니라 **시뮬레이터의 좌표**(`is_in_box`)로 판정합니다.
    GPU, LeRobot, PyTorch는 쓰지 않습니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9-1. 로봇 도구 계층

    LLM은 좌표를 정하고, IK와 관절 보간은 3장 코드가 그대로 합니다. 도구 계층이 하는 일은 세 가지입니다.

    1. **도구 정의(schema)**: 이름, 설명, 인자 형식(JSON Schema). LLM은 이 설명만 보고 도구를 고릅니다.
    2. **안전 장치**: 작업 영역(반지름 0.12~0.30 m, 높이 0.005~0.30 m) 밖의 목표는 잘라내고 경고를 돌려줍니다(3장 3-7의 결과).
       이동 시간도 LLM이 아니라 도구가 거리에 비례해 정합니다(0.15 m/s, 0.8~2초). 모든 이동을 1초로 하면 상자까지의 긴 이동에서 관성으로 블록을 놓칩니다.
    3. **관찰 범위(perception)**: `state`는 정확한 좌표, `color`는 4장 색 검출 추정값(약 1 cm 오차), `camera`는 좌표 없이 `look()`으로 사진만 줍니다.
    """)
    return


@app.cell
def _(base64, io, np, pc):
    R_MIN, R_MAX = 0.12, 0.30
    Z_MIN, Z_MAX = 0.005, 0.30
    MOVE_SPEED = 0.15                 # m/s
    MOVE_MIN, MOVE_MAX = 0.8, 2.0     # s
    SETTLE = 0.2                      # 이동 후 흔들림이 가라앉기를 기다리는 시간

    SYSTEM_PROMPT = """You control a 5-DOF SO-101 robot arm in a MuJoCo simulation.

    Coordinates are in meters in the world frame. The arm base is at the origin.
    The gripper approaches from above: its fingers always point straight down.

    Facts you need:
    - A cube is 3 cm wide and its center sits at z = 0.015 when resting on the table.
    - To grasp a cube, put the gripper at the cube's x, y and z = 0.02, then close it.
      Closing at a higher z grabs nothing; lower z collides with the table.
    - Approach from z = 0.08 above the target before going down. Lift to z = 0.12 before moving sideways,
      otherwise the cube is dragged across the table or knocked over.
    - Release above the box at about z = 0.06.
    - Reachable area: distance from origin between 0.12 and 0.30 m. Targets outside are clamped.

    Reading the result of a move:
    - position_error is how far the gripper ended from the point you asked for, in meters.
      Below 0.02 is fine. Above that, the point is hard to reach with the fingers pointing down.
    - tilted means the arm could not keep the fingers exactly vertical. This is normal near the edge
      of the workspace and does NOT mean the grasp failed. Ignore it unless you are about to grasp.
    - Never re-send a move to the position the gripper actually reached. That is where it already is,
      and repeating it pushes the arm further off. If a point is not reachable, pick a different one,
      usually a lower z, or just carry on with the task.

    To check whether you are holding the cube, call get_scene_state and look at the cube's z.
    A cube on the table sits at z = 0.015; a lifted cube is clearly higher.

    Work one step at a time. When the task is finished or you cannot continue, call report_done.""".replace("\n    ", "\n")


    def _xyz(v):
        return [round(float(x), 4) for x in v]


    def clamp_target(p):
        """작업 영역 밖의 목표를 잘라낸다. (잘라낸 목표, 잘렸는지)"""
        q = p.copy()
        r = float(np.linalg.norm(q[:2]))
        if r < 1e-6:
            q[0], r = R_MIN, R_MIN
        if not R_MIN <= r <= R_MAX:
            q[:2] *= np.clip(r, R_MIN, R_MAX) / r
        q[2] = np.clip(q[2], Z_MIN, Z_MAX)
        return q, bool(not np.allclose(q, p))


    def encode_png(rgb):
        """RGB 배열 → data: URL (Responses API input_image 형식)."""
        import imageio.v3 as iio

        buf = io.BytesIO(iio.imwrite("<bytes>", rgb, extension=".png"))
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


    def _fn(name, desc, props=None, required=None):
        return {"type": "function", "name": name, "description": desc, "strict": True,
                "parameters": {"type": "object", "properties": props or {}, "required": required or [],
                               "additionalProperties": False}}


    class RobotTools:
        """도구 이름 → 실제 동작. perception이 LLM에게 보이는 세상을 정한다."""

        def __init__(self, robot, colors=("red",), perception="state", cam=None):
            self.robot, self.colors, self.perception, self.cam = robot, list(colors), perception, cam
            self.calls = []                 # {"name", "args", "result"} 기록
            self.pending_image = None       # look()이 찍어 둔 사진. 다음 요청에 붙는다

        def schema(self):
            tools = [
                _fn("get_scene_state", "Current gripper pose, gripper open/closed, and what is known about the objects."),
                _fn("move_gripper_to", "Move the gripper to a world position (meters). Fingers stay pointing down.",
                    {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}}, ["x", "y", "z"]),
                _fn("set_gripper", "Open or close the gripper.",
                    {"state": {"type": "string", "enum": ["open", "closed"]}}, ["state"]),
                _fn("report_done", "Call when the task is complete or you cannot continue.",
                    {"success": {"type": "boolean"}, "note": {"type": "string"}}, ["success", "note"]),
            ]
            if self.perception == "camera":
                tools.insert(1, _fn("look", "Take a photo with the overhead camera and look at it."))
            return tools

        def call(self, name, args):
            fn = getattr(self, f"_t_{name}", None)
            if fn is None:
                result = {"error": f"unknown tool {name}"}
            else:
                try:
                    result = fn(**args)
                except TypeError as e:
                    result = {"error": f"bad arguments: {e}"}
            self.calls.append({"name": name, "args": args, "result": result})
            return result

        def _t_get_scene_state(self):
            pos, _ = self.robot.site_pose()
            state = {"gripper_xyz": _xyz(pos),
                     "gripper": "open" if self.robot.gripper_opening() > 0.5 else "closed",
                     "box_xyz": _xyz(self.robot.box_pos())}
            if self.perception == "state":
                state["objects"] = {c: _xyz(self.robot.cube_pos(c)) for c in self.colors}
            elif self.perception == "color":
                seen = {}
                for c in self.colors:
                    p, _ = pc.locate_object(self.cam, self.robot.data, c)
                    if p is not None:
                        seen[c] = _xyz(p)
                state["objects"] = seen
                state["note"] = "positions estimated from the camera, accurate to about 1 cm"
            else:
                state["note"] = "object positions are not available; call look() to see the scene"
            return state

        def _t_look(self):
            rgb, _ = self.cam.capture(self.robot.data)
            self.pending_image = encode_png(rgb)
            return {"image": "attached below", "width": rgb.shape[1], "height": rgb.shape[0],
                    "note": "overhead camera looking straight down at the table"}

        def _t_move_gripper_to(self, x, y, z):
            target, clamped = clamp_target(np.array([float(x), float(y), float(z)]))
            here, _ = self.robot.site_pose()
            seconds = float(np.clip(np.linalg.norm(target - here) / MOVE_SPEED, MOVE_MIN, MOVE_MAX))
            _, err = self.robot.move_to(target, seconds=seconds)
            self.robot.hold(SETTLE)
            pos, _ = self.robot.site_pose()
            # position_error는 실제로 잰 값. tilted는 IK 잔차(위치+방향)가 큰지로 판단하는 대리 지표다.
            out = {"gripper_xyz": _xyz(pos), "position_error": round(float(np.linalg.norm(pos - target)), 4),
                   "tilted": bool(err > 0.1), "seconds": round(seconds, 2)}
            if clamped:
                out["warning"] = f"target was outside the reachable area, clamped to {_xyz(target)}"
            return out

        def _t_set_gripper(self, state):
            if state == "open":
                self.robot.open_gripper(0.5)
            else:
                self.robot.close_gripper(0.8)
            self.robot.hold(SETTLE)
            return {"gripper": state, "opening_rad": round(self.robot.gripper_opening(), 3)}

        def _t_report_done(self, success, note):
            return {"acknowledged": True, "claimed_success": bool(success), "note": note}
    return RobotTools, SYSTEM_PROMPT, clamp_target


@app.cell
def _(RobotTools, SYSTEM_PROMPT, clamp_target, json, mo, np, pc):
    _robot = pc.SO101Sim(render=False)
    _tools = RobotTools(_robot, perception="state")
    _schema = json.dumps(_tools.schema(), indent=1)
    _robot.close()
    _examples = [np.array(p) for p in [(0.24, 0.0, 0.02), (0.40, 0.0, 0.05), (0.05, 0.05, 0.10), (0.2, 0.1, -0.1)]]
    _rows = ["| 요청 | 잘라낸 목표 | 잘렸나 |", "| --- | --- | --- |"] + [
        f"| {p.tolist()} | {np.round(clamp_target(p)[0], 3).tolist()} | {clamp_target(p)[1]} |" for p in _examples]
    mo.accordion({
        "LLM에게 보내는 도구 정의 (schema, perception=state)": mo.md(f"```json\n{_schema}\n```"),
        "시스템 프롬프트": mo.md(f"```text\n{SYSTEM_PROMPT}\n```"),
        "clamp_target 예시": mo.md("\n".join(_rows)),
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 직접 LLM이 되어 보기

    아래에 도구 호출을 한 줄에 하나씩 JSON으로 적고 실행해 보세요. 여러분이 LLM 자리에서 도구를 고르는 것입니다.
    각 호출의 결과(LLM이 받는 JSON)가 표로 나옵니다. 블록은 (0.24, 0)에 있습니다.
    """)
    return


@app.cell
def _(mo):
    manual_calls = mo.ui.text_area(value="\n".join([
        '{"name": "get_scene_state", "args": {}}',
        '{"name": "set_gripper", "args": {"state": "open"}}',
        '{"name": "move_gripper_to", "args": {"x": 0.24, "y": 0.0, "z": 0.08}}',
        '{"name": "move_gripper_to", "args": {"x": 0.24, "y": 0.0, "z": 0.02}}',
        '{"name": "set_gripper", "args": {"state": "closed"}}',
        '{"name": "move_gripper_to", "args": {"x": 0.24, "y": 0.0, "z": 0.12}}',
        '{"name": "get_scene_state", "args": {}}',
    ]), rows=9, full_width=True, label="도구 호출 (JSON Lines)")
    manual_btn = mo.ui.run_button(label="실행")
    mo.vstack([manual_calls, manual_btn])
    return manual_btn, manual_calls


@app.cell
def _(RUN_DIR, RobotTools, json, manual_btn, manual_calls, mo, pc):
    mo.stop(not manual_btn.value, mo.md("호출을 고치고 실행 버튼을 누르세요."))
    _robot = pc.SO101Sim(cube_pos=(0.24, 0.0), box_pos=pc.BOX_XY, render=True, camera="fixed")
    _tools = RobotTools(_robot, perception="state")
    _rows = ["| # | 호출 | 결과 |", "| ---: | --- | --- |"]
    for _i, _line in enumerate(l for l in manual_calls.value.splitlines() if l.strip()):
        try:
            _c = json.loads(_line)
            _res = _tools.call(_c["name"], _c.get("args", {}))
        except (json.JSONDecodeError, KeyError) as _e:
            _res = {"error": f"JSON 형식 오류: {_e}"}
        _rows.append(f"| {_i + 1} | `{_line.strip()}` | `{json.dumps(_res, ensure_ascii=False)}` |")
    _robot.hold(0.3)
    _video = _robot.save_video(RUN_DIR / "manual_tools.mp4")
    _ok = pc.is_in_box(_robot.cube_pos(), _robot.box_pos())
    _robot.close()
    mo.vstack([mo.md("\n".join(_rows) + f"\n\n블록 최종 위치 판정: **{'상자 안 (성공)' if _ok else '상자 밖'}**"),
               mo.video(src=_video, loop=True, width=440) if _video else mo.md("")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9-2. 관찰 → 판단 → 행동 루프

    `run_agent`는 대화 이력(`items`)을 들고 두뇌에게 묻고, 돌아온 도구 호출을 실행하고, 그 결과를 다시 이력에 붙입니다.
    `report_done`이 오거나, 도구를 부르지 않거나, 스텝 상한에 닿으면 멈춥니다.
    두뇌는 `respond(items, tools_schema) → (output_items, 걸린 시간)` 하나만 지키면 됩니다. 아래 세 가지 두뇌가 같은 약속을 따릅니다.
    """)
    return


@app.cell
def _(json):
    def clean_transcript(items):
        """저장 전 정리: base64 이미지를 자리표시자로, 암호화 추론 블록 제거."""
        out = []
        for it in items:
            it = json.loads(json.dumps(it))
            it.pop("encrypted_content", None)
            if isinstance(it.get("content"), list):
                for c in it["content"]:
                    if isinstance(c, dict) and str(c.get("image_url", "")).startswith("data:"):
                        c["image_url"] = f"<image {len(c['image_url'])} chars>"
            out.append(it)
        return out


    def run_agent(client, tools, task, system_prompt, max_steps=25, log=None):
        """도구를 부르는 대화를 끝까지 돌린다. 판정은 하지 않고 진행 기록만 돌려준다."""
        items = [{"role": "system", "content": system_prompt}, {"role": "user", "content": task}]
        stop, claimed, step = "max_steps", None, -1
        for step in range(max_steps):
            output, dt = client.respond(items, tools.schema())
            items += output
            calls = [o for o in output if o.get("type") == "function_call"]
            if not calls:
                stop = "no_call"
                break
            for c in calls:
                args = json.loads(c.get("arguments") or "{}")
                result = tools.call(c["name"], args)
                if log is not None:
                    log.append({"step": step + 1, "tool": c["name"], "args": args, "result": result, "seconds": dt})
                items.append({"type": "function_call_output", "call_id": c["call_id"],
                              "output": json.dumps(result, ensure_ascii=False)})
                if tools.pending_image:                 # look()이 찍은 사진을 다음 요청에 붙인다
                    items.append({"role": "user", "content": [
                        {"type": "input_text", "text": "Here is the photo you asked for."},
                        {"type": "input_image", "image_url": tools.pending_image}]})
                    tools.pending_image = None
                if c["name"] == "report_done":
                    stop, claimed = "done", bool(args.get("success"))
            if stop == "done":
                break
        return {"stop": stop, "claimed_success": claimed, "steps": step + 1,
                "tool_calls": len(tools.calls), "transcript": clean_transcript(items)}
    return (run_agent,)


@app.cell
def _(json, os, re, time):
    class ScriptedPlanner:
        """LLM 대신 쓰는 규칙 기반 두뇌. 대화 이력의 마지막 도구 결과를 읽고 다음 도구를 고른다.

        3장의 상태 기계를 '도구 호출' 형태로 다시 쓴 것이다. 문장에서 색을 정규식으로 뽑으므로
        'Pick up the green cube ...' 같은 정해진 형식만 이해한다. camera 모드(좌표 없음)는 다루지 못한다.
        """

        def __init__(self):
            self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0}

        def respond(self, items, tools):
            self.usage["requests"] += 1
            task = next(i["content"] for i in items if i.get("role") == "user")
            m = re.search(r"the (\w+) cube", task)
            color = m.group(1) if m else "red"
            done = [i for i in items if i.get("type") == "function_call"]
            outputs = [json.loads(i["output"]) for i in items if i.get("type") == "function_call_output"]
            state = next((o for o in reversed(outputs) if "box_xyz" in o), None)   # 가장 최근 get_scene_state 결과
            n = len(done)
            if n == 0:
                return self._call("get_scene_state", {}), 0.0
            objs = (state or {}).get("objects", {})
            if color not in objs:
                return self._call("report_done", {"success": False, "note": f"I cannot locate the {color} cube."}), 0.0
            cx, cy, cz = objs[color]
            bx, by, _ = state["box_xyz"]
            plan = [("set_gripper", {"state": "open"}),
                    ("move_gripper_to", {"x": cx, "y": cy, "z": 0.08}),
                    ("move_gripper_to", {"x": cx, "y": cy, "z": 0.02}),
                    ("set_gripper", {"state": "closed"}),
                    ("move_gripper_to", {"x": cx, "y": cy, "z": 0.12}),
                    ("get_scene_state", {})]
            if n <= len(plan):
                return self._call(*plan[n - 1]), 0.0
            if n == len(plan) + 1 and cz < 0.05:              # 들어 올리지 못했다
                return self._call("report_done", {"success": False, "note": f"The {color} cube was not lifted."}), 0.0
            # 마지막 get_scene_state 이후의 단계 (이 시점의 state는 '들어 올린 뒤')
            box_plan = [("move_gripper_to", {"x": bx, "y": by, "z": 0.12}),
                        ("move_gripper_to", {"x": bx, "y": by, "z": 0.06}),
                        ("set_gripper", {"state": "open"}),
                        ("report_done", {"success": True, "note": f"Placed the {color} cube in the box."})]
            k = n - len(plan) - 1
            return self._call(*box_plan[min(k, len(box_plan) - 1)]), 0.0

        @staticmethod
        def _call(name, args):
            return [{"type": "function_call", "name": name, "arguments": json.dumps(args),
                     "call_id": f"call_{time.time_ns()}"}]


    class ReplayClient:
        """기록된 trace의 응답(도구 호출)을 순서대로 다시 내놓는다. 관측을 보지 않는 '열린 고리' 재생이다."""

        def __init__(self, transcript):
            self.turns, cur = [], []
            for it in transcript:
                if it.get("type") in ("function_call", "message", "reasoning"):
                    cur.append(it)
                elif cur:
                    self.turns.append(cur)
                    cur = []
            if cur:
                self.turns.append(cur)
            self.i = 0
            self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0}

        def respond(self, items, tools):
            self.usage["requests"] += 1
            if self.i >= len(self.turns):
                return [], 0.0
            out = [dict(o) for o in self.turns[self.i] if o.get("type") == "function_call"]
            self.i += 1
            return out, 0.0


    class OpenAIClient:
        """Responses API를 부르는 얇은 래퍼. 접속 정보는 환경변수(OPENAI_API_KEY 등)에서 읽는다."""

        def __init__(self, model=None, timeout=120.0):
            from openai import OpenAI

            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("환경변수 OPENAI_API_KEY가 없습니다.")
            self.client = OpenAI(timeout=timeout)
            self.model = model or os.getenv("OPENAI_MODEL") or "gpt-6-astra"
            self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0}

        def respond(self, input_items, tools):
            last = None
            for attempt in range(3):
                t0 = time.time()
                try:
                    r = self.client.responses.create(model=self.model, input=input_items, tools=tools)
                except Exception as e:  # noqa: BLE001  네트워크·일시적 오류는 잠시 쉬고 재시도
                    last = e
                    time.sleep(2 * (attempt + 1))
                    continue
                dt = time.time() - t0
                self.usage["requests"] += 1
                self.usage["seconds"] += dt
                if getattr(r, "usage", None) is not None:
                    self.usage["input_tokens"] += r.usage.input_tokens or 0
                    self.usage["output_tokens"] += r.usage.output_tokens or 0
                return [o.model_dump(exclude_none=True) for o in r.output], dt
            raise RuntimeError(f"LLM 호출이 3회 실패했습니다: {last}")
    return OpenAIClient, ReplayClient, ScriptedPlanner


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9-3. LLM이 지휘하는 Pick and Place

    3장과 같은 과제, 같은 블록 위치, 같은 성공 판정입니다. 다른 것은 **다섯 단계의 순서를 사람이 적었느냐, 두뇌가 정하느냐** 하나입니다.

    - `replay`를 고르면 원본 trace의 블록 위치를 그대로 쓰고, 저자가 기록한 LLM의 도구 호출을 그대로 재생합니다.
      재생은 관측을 다시 보지 않으므로, 우리 시뮬레이터(MuJoCo·모델 버전)에서 결과가 기록과 달라질 수 있습니다.
    - `openai`는 매 호출마다 요금이 발생합니다. 원본 측정에서 한 Episode에 약 12회 호출, 1.3만 토큰이었습니다.
    """)
    return


@app.cell
def _(mo):
    trace_dl_btn = mo.ui.run_button(label="원본 LLM 실행 기록(trace) 내려받기 (약 90 KB)")
    trace_dl_btn
    return (trace_dl_btn,)


@app.cell
def _(Path, mo, os, pc, trace_dl_btn):
    if trace_dl_btn.value:
        with mo.status.spinner("내려받는 중..."):
            pc.download_release("ch09_traces")
    trace_dir = pc.CHECKPOINT_DIR / "ch09_traces" / "ch09_traces"
    trace_files = sorted(p.name for p in trace_dir.glob("*.json")) if trace_dir.is_dir() else []
    _has_key = bool(os.getenv("OPENAI_API_KEY"))
    try:
        import openai  # noqa: F401

        _has_openai = True
    except ImportError:
        _has_openai = False
    mo.md(f"- trace: {'있음 (' + str(len(trace_files)) + '개)' if trace_files else '없음'} "
          f"`{Path(trace_dir).relative_to(pc.REPO_ROOT)}`\n"
          f"- openai 패키지: {'설치됨' if _has_openai else '없음 (`uv sync --locked --extra llm`)'}, "
          f"OPENAI_API_KEY: {'설정됨' if _has_key else '없음'}")
    return trace_dir, trace_files


@app.cell
def _(mo, trace_files):
    pick_brain = mo.ui.dropdown(["scripted", "replay", "openai"], value="scripted", label="두뇌")
    pick_perception = mo.ui.dropdown(["state", "color", "camera"], value="state", label="perception")
    _pick_traces = [f for f in trace_files if f.startswith(("showcase", "ch09_pick", "repeat"))]
    pick_trace = mo.ui.dropdown(_pick_traces or ["(trace 없음)"], value=(_pick_traces or ["(trace 없음)"])[0],
                                label="replay할 trace")
    pick_episode = mo.ui.number(value=0, start=0, label="trace의 Episode 번호")
    pick_model = mo.ui.text(value="", placeholder="비우면 OPENAI_MODEL 또는 gpt-6-astra", label="openai 모델")
    pick_btn = mo.ui.run_button(label="실행")
    mo.vstack([mo.hstack([pick_brain, pick_perception, pick_btn], justify="start"),
               mo.hstack([pick_trace, pick_episode, pick_model], justify="start")])
    return (
        pick_brain,
        pick_btn,
        pick_episode,
        pick_model,
        pick_perception,
        pick_trace,
    )


@app.cell
def _(
    OpenAIClient,
    RUN_DIR,
    ReplayClient,
    RobotTools,
    SYSTEM_PROMPT,
    ScriptedPlanner,
    json,
    mo,
    pc,
    pick_brain,
    pick_btn,
    pick_episode,
    pick_model,
    pick_perception,
    pick_trace,
    run_agent,
    trace_dir,
):
    mo.stop(not pick_btn.value, mo.md("두뇌를 고르고 실행 버튼을 누르세요."))
    _cube, _perception = (0.24, 0.0), pick_perception.value
    if pick_brain.value == "replay":
        mo.stop(not (trace_dir / pick_trace.value).is_file(), mo.md("trace가 없습니다. 위에서 내려받으세요."))
        _d = json.loads((trace_dir / pick_trace.value).read_text())
        _ep = _d["episodes"][int(pick_episode.value)]
        mo.stop("transcript" not in _ep, mo.md("이 Episode에는 대화 기록이 없습니다. 다른 번호를 고르세요."))
        _cube, _perception = tuple(_ep["cube_start"][:2]), _d["perception"]
        _client = ReplayClient(_ep["transcript"])
        _recorded = f"기록된 결과: {'성공' if _ep.get('success') else '실패'}, 도구 호출 {_ep['tool_calls']}회 ({_d['model']}, {_d['perception']})"
    elif pick_brain.value == "openai":
        try:
            _client = OpenAIClient(model=pick_model.value or None)
        except Exception as _e:  # noqa: BLE001
            mo.stop(True, mo.callout(mo.md(f"OpenAI 클라이언트를 만들 수 없습니다: {_e}"), kind="danger"))
        _recorded = f"모델 `{_client.model}`"
    else:
        mo.stop(_perception == "camera", mo.md("scripted 두뇌는 사진을 볼 수 없습니다. perception을 state나 color로 고르세요."))
        _client, _recorded = ScriptedPlanner(), "규칙 기반 Planner"
    _robot = pc.SO101Sim(cube_pos=_cube, box_pos=pc.BOX_XY, render=True, camera="fixed")
    _cam = pc.SimCamera(_robot.model, "top", 320, 240) if _perception in ("color", "camera") else None
    _tools = RobotTools(_robot, colors=["red"], perception=_perception, cam=_cam)
    _log = []
    with mo.status.spinner("에이전트 실행 중..."):
        _rec = run_agent(_client, _tools, pc.TASK, SYSTEM_PROMPT, max_steps=25, log=_log)
    _success = bool(pc.is_in_box(_robot.cube_pos(), _robot.box_pos()))
    _robot.hold(0.3)
    _video = _robot.save_video(RUN_DIR / f"llm_pick_{pick_brain.value}.mp4")
    if _cam is not None:
        _cam.close()
    _robot.close()
    _rec.update({"brain": pick_brain.value, "perception": _perception, "cube_start": list(_cube),
                 "success": _success, "usage": _client.usage})
    (RUN_DIR / f"llm_pick_{pick_brain.value}.json").write_text(json.dumps(_rec, indent=1, ensure_ascii=False))
    _rows = ["| # | 도구 | 인자 | 결과 |", "| ---: | --- | --- | --- |"] + [
        f"| {l['step']} | `{l['tool']}` | `{json.dumps(l['args'])}` | `{json.dumps(l['result'], ensure_ascii=False)[:160]}` |"
        for l in _log]
    mo.vstack([
        mo.md(f"### 판정: {'성공' if _success else '실패'} (두뇌의 주장: {_rec['claimed_success']}, 종료 이유 `{_rec['stop']}`)\n\n"
              f"블록 시작 위치 {_cube}, perception `{_perception}`, {_recorded}. "
              f"LLM 호출 {_client.usage['requests']}회, 토큰 {_client.usage['input_tokens'] + _client.usage['output_tokens']:,}"),
        mo.hstack([mo.video(src=_video, loop=True, width=420) if _video else mo.md(""), mo.md("\n".join(_rows))], align="start"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **관찰 포인트**

    - 들어 올린 뒤 `position_error`가 크고 `tilted: true`가 나와도 블록은 잡혀 있을 수 있습니다. 작업 영역 가장자리에서
      손가락을 수직으로 유지하지 못할 뿐입니다. 시스템 프롬프트는 이 경우를 무시하라고 알려 줍니다.
    - 원본 책에서는 도구 출력과 지침을 고치기 전 성공률이 4/10이었고, 고친 뒤 9/10이 되었습니다(trace `ch09_pick_state` vs `ch09_pick_v3_state`).
      LLM Planner의 성능은 모델만큼이나 **도구 설계와 설명 문장**에 달려 있습니다.

    ## 9-4. 8장의 색 지정 과제를 LLM Planner로

    같은 블록 배치(`random_layout`, seed 2000), 같은 문장, 같은 판정으로 비교합니다.
    "건드린 블록"은 1 cm 넘게 움직인 블록 중 가장 많이 움직인 것입니다(8장의 '손끝이 가장 가까이 간 블록'과는 재는 방법이 다릅니다).
    """)
    return


@app.cell
def _(mo):
    mc_brain = mo.ui.dropdown(["scripted", "openai"], value="scripted", label="두뇌")
    mc_perception = mo.ui.dropdown(["state", "color", "camera"], value="state", label="perception")
    mc_layouts = mo.ui.slider(1, 20, value=3, label="배치 수 (배치당 3색)")
    mc_model = mo.ui.text(value="", placeholder="openai 모델 (선택)", label="모델")
    mc_btn = mo.ui.run_button(label="색 지정 과제 실행")
    mo.hstack([mc_brain, mc_perception, mc_layouts, mc_model, mc_btn], justify="start")
    return mc_brain, mc_btn, mc_layouts, mc_model, mc_perception


@app.cell
def _(
    OpenAIClient,
    RUN_DIR,
    RobotTools,
    SYSTEM_PROMPT,
    ScriptedPlanner,
    json,
    mc_brain,
    mc_btn,
    mc_layouts,
    mc_model,
    mc_perception,
    mo,
    np,
    pc,
    run_agent,
):
    mo.stop(not mc_btn.value, mo.md("버튼을 누르면 실행합니다. scripted는 배치당 약 10초, openai는 배치당 수 분과 요금이 듭니다."))
    mo.stop(mc_brain.value == "scripted" and mc_perception.value == "camera", mo.md("scripted는 camera 모드를 지원하지 않습니다."))
    _client = OpenAIClient(model=mc_model.value or None) if mc_brain.value == "openai" else ScriptedPlanner()
    _rng = np.random.default_rng(2000)
    _stats = {c: {"성공": 0, "오답": 0, "실패": 0} for c in pc.COLORS}
    _records = []
    with mo.status.progress_bar(total=mc_layouts.value * 3, title="LLM Planner 색 지정") as _bar:
        for _L in range(mc_layouts.value):
            _layout = pc.random_layout(_rng)
            for _color in pc.COLORS:
                _robot = pc.SO101Sim(cube_pos=_layout[0], box_pos=pc.BOX_XY, render=False,
                                     extra_cubes=[("green", _layout[1]), ("yellow", _layout[2])])
                _cam = pc.SimCamera(_robot.model, "top", 320, 240) if mc_perception.value != "state" else None
                _tools = RobotTools(_robot, colors=pc.COLORS, perception=mc_perception.value, cam=_cam)
                _start = {c: _robot.cube_pos(c).copy() for c in pc.COLORS}
                _rec = run_agent(_client, _tools, pc.TASK_FMT.format(color=_color), SYSTEM_PROMPT, max_steps=30)
                _in_box = [c for c in pc.COLORS if pc.is_in_box(_robot.cube_pos(c), _robot.box_pos())]
                _result = "성공" if _in_box == [_color] else (f"오답({','.join(_in_box)})" if _in_box else "실패")
                _moved = {c: float(np.linalg.norm(_robot.cube_pos(c) - _start[c])) for c in pc.COLORS}
                _touched = max(_moved, key=_moved.get) if max(_moved.values()) > 0.01 else None
                _stats[_color][_result.split("(")[0]] += 1
                _rec.pop("transcript", None)
                _records.append({"layout": _L, "color": _color, "result": _result, "touched": _touched, **_rec})
                if _cam is not None:
                    _cam.close()
                _robot.close()
                _bar.update()
    (RUN_DIR / f"llm_multicolor_{mc_brain.value}_{mc_perception.value}.json").write_text(json.dumps(
        {"brain": mc_brain.value, "perception": mc_perception.value, "usage": _client.usage, "stats": _stats,
         "episodes": _records}, indent=1, ensure_ascii=False))
    _n = len(_records)
    _ok = sum(s["성공"] for s in _stats.values())
    _match = sum(r["touched"] == r["color"] for r in _records)
    _rows = ["| 명령 색 | 성공 | 오답 | 실패 |", "| --- | ---: | ---: | ---: |"] + [
        f"| {c} | {s['성공']} | {s['오답']} | {s['실패']} |" for c, s in _stats.items()]
    mo.md("\n".join(_rows) + f"\n\n전체 성공률 **{_ok}/{_n}**, 명령한 색의 블록을 건드린 비율 **{_match}/{_n}**. "
          f"LLM 호출 {_client.usage['requests']}회, 토큰 {_client.usage['input_tokens'] + _client.usage['output_tokens']:,}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `scripted` + `state`가 거의 항상 성공하는 것은 놀랍지 않습니다. 정확한 좌표가 주어지면 3장의 규칙으로 충분하기 때문입니다.
    LLM Planner의 가치는 **문장 형식이 정해져 있지 않거나**("왼쪽에 있는 것 말고 노란 것"), **좌표 없이 사진만 있을 때**(`camera`)
    드러납니다. 그 대가는 호출당 수 초의 지연, 토큰 요금, 그리고 매번 조금씩 달라질 수 있는 결정입니다.

    ## 9-5. 원본 실행 기록 읽기

    원본 책의 측정값을 trace에서 직접 집계합니다(`usage.seconds`는 API 응답을 기다린 시간의 합계).
    """)
    return


@app.cell
def _(json, mo, trace_dir, trace_files):
    mo.stop(not trace_files, mo.md("9-3에서 trace를 내려받으면 여기에 요약이 나옵니다."))
    _rows = ["| 파일 | 모델 | perception | 성공 | 평균 도구 호출 | 총 토큰 | API 대기 |", "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for _f in trace_files:
        _d = json.loads((trace_dir / _f).read_text())
        _eps = _d["episodes"]
        _succ = sum(1 for e in _eps if e.get("success") or e.get("result") == "성공")
        _u = _d.get("usage", {})
        _rows.append(f"| `{_f}` | {_d.get('model')} | {_d.get('perception')} | {_succ}/{len(_eps)} | "
                     f"{sum(e.get('tool_calls', 0) for e in _eps) / len(_eps):.1f} | "
                     f"{_u.get('input_tokens', 0) + _u.get('output_tokens', 0):,} | {_u.get('seconds', 0):.0f} s |")
    mo.md("\n".join(_rows))
    return


@app.cell
def _(mo, trace_files):
    view_file = mo.ui.dropdown(trace_files or ["(없음)"], value=("showcase_camera.json" if "showcase_camera.json" in trace_files
                                                               else (trace_files or ["(없음)"])[0]), label="대화 보기")
    view_ep = mo.ui.number(value=0, start=0, label="Episode")
    mo.hstack([view_file, view_ep], justify="start")
    return view_ep, view_file


@app.cell
def _(json, mo, trace_dir, trace_files, view_ep, view_file):
    mo.stop(not trace_files)
    _d = json.loads((trace_dir / view_file.value).read_text())
    _eps = _d["episodes"]
    _ep = _eps[min(int(view_ep.value), len(_eps) - 1)]
    mo.stop("transcript" not in _ep, mo.md("이 Episode에는 대화 전문이 없습니다."))
    _lines = []
    for _it in _ep["transcript"]:
        _t = _it.get("type") or _it.get("role")
        if _t == "system":
            _lines.append("**system**: (시스템 프롬프트 생략)")
        elif _t == "user":
            _c = _it["content"]
            _lines.append(f"**user**: {_c if isinstance(_c, str) else ' '.join(x.get('text', x.get('image_url', '')) for x in _c)}")
        elif _t == "function_call":
            _lines.append(f"**→ {_it['name']}** `{_it.get('arguments')}`")
        elif _t == "function_call_output":
            _lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;← `{_it['output'][:200]}`")
        elif _t == "message":
            _lines.append(f"**assistant**: {' '.join(c.get('text', '') for c in _it.get('content', []))}")
    _meta = {k: v for k, v in _ep.items() if k != "transcript"}
    mo.vstack([mo.md(f"`{json.dumps(_meta, ensure_ascii=False)}`"), mo.md("\n\n".join(_lines))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 정리

    | | Rule 기반 (3·4장) | 강화학습 (5장) | 모방학습 ACT (6장) | VLA SmolVLA (7·8장) | LLM Planner (9장) |
    | --- | --- | --- | --- | --- | --- |
    | 사람이 주는 것 | 단계와 좌표 계산 | 보상 함수 | 시연 데이터 | 시연 + 문장 | 도구와 설명 |
    | 언어 이해 | 없음 | 없음 | 없음 | 학습 데이터 범위 | 넓음 |
    | 제어 주기 | 500 Hz | 50 Hz | 20 Hz | 20 Hz (묶음) | 0.2~2 Hz (도구 단위) |
    | 약점 | 인식 오차, 예외 상황 | 보상 설계, 긴 과제 | 데이터 분포 밖 | 데이터·연산량 | 지연, 요금, 정밀 제어 불가 |

    실제 시스템은 이 층을 섞어 씁니다. LLM이 무엇을 할지 정하고, 학습된 정책이나 IK가 어떻게 움직일지를 맡습니다.

    **확인 문제**

    1. `MOVE_MIN`을 0.3초로 줄이면 LLM Planner의 성공률은 어떻게 될까요? 왜 이동 시간을 LLM이 정하게 하지 않았을까요?
    2. `ScriptedPlanner`가 "Put the yellow one into the box"라는 문장을 받으면 어떻게 되나요? LLM은 어떨까요?
    """)
    return


if __name__ == "__main__":
    app.run()
