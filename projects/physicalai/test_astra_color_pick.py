"""Camera-based Astra object selection followed by SO-101 IK pick-and-place.

prepare: render anonymously labelled scenes and a structured model request.
execute: validate model selections, then run each command from a fresh reset.
The planner receives images and commands, never the simulator's color-to-ID map.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from record_demos import in_box, pick_and_place
from so101_sim import SO101Sim
from test_color_pick import COLORS, VIDEO_FPS, label, random_layout


COMMANDS = [
    ('red', '빨간색 블록만 파란 상자에 옮겨 주세요.'),
    ('green', '초록색 블록만 파란 상자에 옮겨 주세요.'),
    ('yellow', '노란색 블록만 파란 상자에 옮겨 주세요.'),
    ('purple', '보라색 블록만 파란 상자에 옮겨 주세요.'),
]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def project(robot, point, width=640, height=480):
    cam = robot.model.camera('top').id
    rotation = robot.data.cam_xmat[cam].reshape(3, 3)
    x, y, z = (np.asarray(point) - robot.data.cam_xpos[cam]) @ rotation
    focal = .5 * height / np.tan(np.deg2rad(robot.model.cam_fovy[cam]) / 2)
    assert z < 0
    return int(width / 2 + focal * x / -z), int(height / 2 - focal * y / -z)


def prepare(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'scenes.json').exists(), 'Do not overwrite prepared scenes'
    robot = SO101Sim(colors=COLORS)
    renderer = mujoco.Renderer(robot.model, height=480, width=640)
    rng = np.random.default_rng(args.seed)
    scenes = []
    try:
        for scene in range(args.layouts):
            layout = random_layout(rng, COLORS)
            ids = rng.permutation(['A', 'B', 'C', 'D']).tolist()
            mapping = dict(zip(ids, COLORS))
            robot.reset(layout)
            robot.hold(.3)
            renderer.update_scene(robot.data, camera='top')
            frame = renderer.render().copy()
            imageio.imwrite(out / f'scene-{scene:02d}-raw.png', frame)
            for oid, color in mapping.items():
                x, y = project(robot, robot.cube_pos(color))
                cv2.line(frame, (x, y - 8), (x, y - 20), (0, 0, 0), 1)
                for thickness, fg in [(4, (255, 255, 255)), (2, (0, 0, 0))]:
                    cv2.putText(frame, oid, (x - 8, y - 25), cv2.FONT_HERSHEY_SIMPLEX,
                                .65, fg, thickness, cv2.LINE_AA)
            imageio.imwrite(out / f'scene-{scene:02d}.png', frame)
            scenes.append(dict(scene=scene, image=f'scene-{scene:02d}.png',
                               layout=layout, object_id_to_color=mapping))
    finally:
        renderer.close()
    write(out / 'scenes.json', scenes)
    schema = dict(type='object', properties={
        'selections': dict(type='array', items=dict(type='object', properties={
            'scene': {'type': 'integer'}, 'instruction': {'type': 'integer'},
            'object_id': {'type': 'string', 'enum': ['A', 'B', 'C', 'D', 'NONE']},
            'reason': {'type': 'string'},
        }, required=['scene', 'instruction', 'object_id', 'reason'], additionalProperties=False)),
    }, required=['selections'], additionalProperties=False)
    write(out / 'selection-schema.json', schema)
    instructions = '\n'.join(f'{i}: {command}' for i, (_, command) in enumerate(COMMANDS))
    prompt = f'''You are the visual target selector of a simulated robot.
Use only the attached images. Do not call tools or read any files.
There are {args.layouts} images in order: scene 0 through scene {args.layouts - 1}.
Each scene contains four colored cubes labelled A, B, C, D, plus a blue destination box.
Labels are anonymous and are independently reassigned in every scene.
For EACH scene, answer EACH of the following Korean instructions independently.
Select the object ID of the cube to move. Do not select the destination box.
If the requested cube cannot be identified, use NONE. Explain your visual selection briefly.
Instructions:
{instructions}
Return exactly {args.layouts * len(COMMANDS)} entries with scene and instruction indices.
Return only the JSON object matching the provided schema. Do not move anything yourself.
'''
    (out / 'planner-prompt.txt').write_text(prompt)
    print(f'Prepared {len(scenes)} scenes in {out}', flush=True)


def execute(args):
    out = args.out.resolve()
    assert not (out / 'results.json').exists(), 'Do not overwrite completed results'
    scenes = json.loads((out / 'scenes.json').read_text())
    raw = json.loads((out / 'planner-selections.json').read_text())
    selections = raw['selections']
    pairs = [(s['scene'], s['instruction']) for s in selections]
    expected = {(s['scene'], i) for s in scenes for i in range(len(COMMANDS))}
    assert len(pairs) == len(set(pairs)) and set(pairs) == expected, 'Missing or repeated model decisions'
    assert all(s['object_id'] in ['A', 'B', 'C', 'D', 'NONE'] for s in selections)
    lookup = {(s['scene'], s['instruction']): s for s in selections}
    robot = SO101Sim(colors=COLORS)
    renderer = mujoco.Renderer(robot.model, height=480, width=640)
    render_every = round(1 / (VIDEO_FPS * robot.model.opt.timestep))
    (out / 'videos').mkdir(exist_ok=True)
    results = []
    combined = imageio.get_writer(out / 'all_trials.mp4', fps=VIDEO_FPS)
    try:
        for scene in scenes:
            for i, (target, command) in enumerate(COMMANDS):
                decision = lookup[scene['scene'], i]
                selected = scene['object_id_to_color'].get(decision['object_id'])
                robot.on_step = None
                robot.reset(scene['layout'])
                robot.hold(.3)
                before = {c: robot.cube_pos(c).copy() for c in COLORS}
                frames = []
                caption = f"scene {scene['scene']}  command={target}  selected={selected}"

                def capture(sim):
                    if sim.n_steps % render_every == 0:
                        renderer.update_scene(sim.data, camera='view')
                        frames.append(label(renderer.render().copy(), caption, target))

                robot.on_step = capture
                robot.hold(.5)
                skill_success = bool(pick_and_place(robot, selected)) if selected else False
                robot.hold(.5)
                placed = [c for c in COLORS if in_box(robot, c)]
                selection_correct = selected == target
                success = selection_correct and placed == [target]
                outcome = 'SUCCESS' if success else ('WRONG CUBE' if placed else 'FAIL')
                caption = f"scene {scene['scene']}  command={target}  {outcome}"
                robot.hold(.8)
                video = f"videos/scene-{scene['scene']:02d}-cmd-{target}.mp4"
                imageio.mimsave(out / video, frames, fps=VIDEO_FPS)
                for frame in frames:
                    combined.append_data(frame)
                record = dict(scene=scene['scene'], instruction=i, command=command,
                    target=target, selected_object_id=decision['object_id'], selected_color=selected,
                    model_reason=decision['reason'], selection_correct=selection_correct,
                    skill_returned_success=skill_success, placed=placed, success=success,
                    start_xyz={c: before[c].tolist() for c in COLORS},
                    final_xyz={c: robot.cube_pos(c).tolist() for c in COLORS},
                    displacement={c: float(np.linalg.norm(robot.cube_pos(c)-before[c])) for c in COLORS},
                    video=video)
                results.append(record)
                write(out / 'partial-results.json', results)
                print(json.dumps({k: record[k] for k in ['scene', 'target', 'selected_color',
                                                       'selection_correct', 'placed', 'success']}), flush=True)
    finally:
        combined.close()
        renderer.close()
        robot.on_step = None
    summary = dict(episodes=len(results), selection_correct=sum(r['selection_correct'] for r in results),
                   success=sum(r['success'] for r in results),
                   wrong_cube=sum(bool(r['placed']) and r['placed'] != [r['target']] for r in results),
                   no_placement=sum(not r['placed'] for r in results))
    summary['by_color'] = {c: dict(episodes=sum(r['target'] == c for r in results),
                                  success=sum(r['target'] == c and r['success'] for r in results))
                           for c in COLORS}
    scene_hashes = {s['image']: hashlib.sha256((out/s['image']).read_bytes()).hexdigest() for s in scenes}
    write(out / 'results.json', dict(model='gpt-6.2-astra', backend='codex exec',
        control='model image-based object ID selection + existing IK pick_and_place',
        model_inputs='anonymous labelled scene images and Korean instructions only',
        seed=args.seed, scene_hashes=scene_hashes, summary=summary, episodes=results))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'execute'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--layouts', type=int, default=3)
    p.add_argument('--seed', type=int, default=62)
    args = p.parse_args()
    (prepare if args.stage == 'prepare' else execute)(args)
