"""
    Spot Collision Tuner:

    Interactive tool for fitting collision primitives to the Spot model. Click
    a primitive, drag it in the view plane, nudge/rotate/resize it from the
    keyboard, and write the result back into an MJCF file. The model is held
    kinematically (no physics) with every joint at 0 by default, or at its
    `home` keyframe (stowed arm) with `--pose home`.

        uv run python -m training.envs.spot.tune_collision

    Mouse: left-drag orbit, right-drag pan, scroll zoom, click to select,
    shift+left-drag moves the selected geom, shift+right-drag rotates it.
"""
import argparse
import re
import shutil
import time
from pathlib import Path

import glfw
import mujoco
import numpy as np

_MJCF_DIR = Path(__file__).resolve().parent / 'mjcf'
_DEFAULT_SCENE = _MJCF_DIR / 'scene_mjx_simplified.xml'
# Never default to the base (true-mesh) models: tuning writes to these files.
_DEFAULT_XMLS = (
    _MJCF_DIR / 'models' / 'spot_simplified.xml',
    _MJCF_DIR / 'models' / 'arm_simplified.xml',
)

_GEOM = mujoco.mjtGeom
# Leading `geom_size` entries each tunable primitive type actually uses.
_SIZE_COUNT = {
    int(_GEOM.mjGEOM_SPHERE): 1,
    int(_GEOM.mjGEOM_CAPSULE): 2,
    int(_GEOM.mjGEOM_CYLINDER): 2,
    int(_GEOM.mjGEOM_ELLIPSOID): 3,
    int(_GEOM.mjGEOM_BOX): 3,
}
_MIN_SIZE = 0.001
# Runtime-only geom group for tunables, so picking can target just them.
TUNE_GROUP = 4


def find_tunable_geoms(model: mujoco.MjModel, pattern: str | None = None) -> list[int]:
    """Returns ids of named collision primitives (group 3) to tune.

    Args:
        model: Compiled model.
        pattern: Regex on geom names. When omitted, every primitive is
            returned except feet (their radius is a sourced value).
    """
    geom_ids = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if not name or model.geom_group[geom_id] != 3:
            continue
        if int(model.geom_type[geom_id]) not in _SIZE_COUNT:
            continue
        if (re.search(pattern, name) if pattern else 'foot' not in name):
            geom_ids.append(geom_id)
    return geom_ids


def _bounding_radius(geom_type: int, size: np.ndarray) -> float:
    if geom_type == int(_GEOM.mjGEOM_CAPSULE):
        return float(size[0] + size[1])
    if geom_type == int(_GEOM.mjGEOM_CYLINDER):
        return float(np.hypot(size[0], size[1]))
    if geom_type == int(_GEOM.mjGEOM_SPHERE):
        return float(size[0])
    return float(np.linalg.norm(size))


class CollisionTuner:
    """Editable pose and size for a set of collision primitives.

    Edits write straight into `model.geom_pos/quat/size`, so the stock
    MuJoCo renderer shows them. Poses are in each geom's parent-body frame,
    matching the numbers in the MJCF.

    Attributes:
        model: Model being edited in place.
        data: Data at the fixed pose the fit is judged in.
        geom_ids: Geoms that can be selected.
        index: Position of the selected geom within `geom_ids`.
    """

    def __init__(self, model, data, geom_ids: list[int]):
        if not geom_ids:
            raise ValueError('No tunable collision primitives found.')
        self.model = model
        self.data = data
        self.geom_ids = list(geom_ids)
        self.index = 0
        self._original = {
            g: (model.geom_pos[g].copy(), model.geom_quat[g].copy(), model.geom_size[g].copy())
            for g in self.geom_ids
        }

    @property
    def geom_id(self) -> int:
        """Id of the selected geom."""
        return self.geom_ids[self.index]

    def name(self, geom_id: int) -> str:
        """Returns the geom's MJCF name."""
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)

    def select(self, geom_id: int) -> bool:
        """Selects `geom_id` if tunable; returns whether it was."""
        if geom_id not in self.geom_ids:
            return False
        self.index = self.geom_ids.index(geom_id)
        return True

    def cycle(self, step: int) -> None:
        """Moves the selection by `step` (wraps)."""
        self.index = (self.index + step) % len(self.geom_ids)

    def _update(self, geom_id: int) -> None:
        # Culling uses the bounding radius, so keep it valid as sizes change.
        model = self.model
        model.geom_rbound[geom_id] = _bounding_radius(
            int(model.geom_type[geom_id]), model.geom_size[geom_id])
        mujoco.mj_kinematics(model, self.data)

    def nudge(self, axis: int, amount: float) -> None:
        """Translates the selected geom along a parent-frame axis [m]."""
        self.model.geom_pos[self.geom_id, axis] += amount
        self._update(self.geom_id)

    def resize(self, index: int, amount: float) -> None:
        """Grows `geom_size[index]` by `amount` [m], if the type uses it."""
        geom_id = self.geom_id
        if index >= _SIZE_COUNT[int(self.model.geom_type[geom_id])]:
            return
        size = self.model.geom_size[geom_id]
        size[index] = max(size[index] + amount, _MIN_SIZE)
        self._update(geom_id)

    def _rotate_parent(self, delta_quat: np.ndarray) -> None:
        geom_id = self.geom_id
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, delta_quat, self.model.geom_quat[geom_id])
        mujoco.mju_normalize4(quat)
        self.model.geom_quat[geom_id] = quat
        self._update(geom_id)

    def rotate(self, axis: int, radians: float) -> None:
        """Rotates the selected geom about a parent-frame axis, in place."""
        unit = np.zeros(3)
        unit[axis] = 1.0
        delta = np.zeros(4)
        mujoco.mju_axisAngle2Quat(delta, unit, radians)
        self._rotate_parent(delta)

    def drag_in_view(self, dx_px: float, dy_px: float, height_px: float, camera) -> None:
        """Moves the selected geom parallel to the image plane.

        Args:
            dx_px: Mouse motion right [px].
            dy_px: Mouse motion down [px].
            height_px: Framebuffer height [px].
            camera: `MjvGLCamera` of the last rendered scene.
        """
        geom_id = self.geom_id
        forward, up = np.array(camera.forward), np.array(camera.up)
        right = np.cross(forward, up)
        depth = float((self.data.geom_xpos[geom_id] - camera.pos) @ forward)
        frustum_height = camera.frustum_top - camera.frustum_bottom
        meters_per_px = frustum_height * depth / camera.frustum_near / height_px
        delta_world = (right * dx_px - up * dy_px) * meters_per_px
        parent = self.data.xmat[self.model.geom_bodyid[geom_id]].reshape(3, 3)
        self.model.geom_pos[geom_id] += parent.T @ delta_world
        self._update(geom_id)

    def rotate_in_view(self, dx_px: float, dy_px: float, camera, radians_per_px: float = 0.01) -> None:
        """Trackball-rotates the selected geom about its own center."""
        forward, up = np.array(camera.forward), np.array(camera.up)
        right = np.cross(forward, up)
        about_up, about_right = np.zeros(4), np.zeros(4)
        mujoco.mju_axisAngle2Quat(about_up, up, dx_px * radians_per_px)
        mujoco.mju_axisAngle2Quat(about_right, right, dy_px * radians_per_px)
        world = np.zeros(4)
        mujoco.mju_mulQuat(world, about_up, about_right)
        # Express the world-frame rotation in the parent body's frame.
        body = self.data.xquat[self.model.geom_bodyid[self.geom_id]].copy()
        body_inv, tmp, parent = np.zeros(4), np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(body_inv, body)
        mujoco.mju_mulQuat(tmp, body_inv, world)
        mujoco.mju_mulQuat(parent, tmp, body)
        self._rotate_parent(parent)

    def reset(self) -> None:
        """Restores the selected geom to its loaded pose and size."""
        geom_id = self.geom_id
        pos, quat, size = self._original[geom_id]
        self.model.geom_pos[geom_id] = pos
        self.model.geom_quat[geom_id] = quat
        self.model.geom_size[geom_id] = size
        self._update(geom_id)

    def is_modified(self, geom_id: int) -> bool:
        """Returns whether `geom_id` differs from its loaded values."""
        pos, quat, size = self._original[geom_id]
        model = self.model
        return not (
            np.allclose(model.geom_pos[geom_id], pos, atol=1e-9)
            and np.allclose(model.geom_size[geom_id], size, atol=1e-9)
            # q and -q are the same rotation.
            and abs(float(model.geom_quat[geom_id] @ quat)) > 1 - 1e-9
        )

    def modified_ids(self) -> list[int]:
        """Returns ids of all geoms edited so far."""
        return [g for g in self.geom_ids if self.is_modified(g)]

    def commit(self) -> None:
        """Treats current values as the loaded ones (call after saving)."""
        m = self.model
        self._original = {
            g: (m.geom_pos[g].copy(), m.geom_quat[g].copy(), m.geom_size[g].copy())
            for g in self.geom_ids
        }


def _format(values) -> str:
    return ' '.join(f'{0.0 if abs(v) < 1e-9 else v:.6g}' for v in values)


def geom_attributes(model: mujoco.MjModel, geom_id: int) -> dict[str, str]:
    """Returns the MJCF `pos`/`quat`/`size` strings for a geom."""
    quat = model.geom_quat[geom_id].copy()
    mujoco.mju_normalize4(quat)
    if quat[0] < 0:
        quat = -quat
    count = _SIZE_COUNT[int(model.geom_type[geom_id])]
    return {
        'pos': _format(model.geom_pos[geom_id]),
        'quat': _format(quat),
        'size': _format(model.geom_size[geom_id][:count]),
    }


def patch_geom_xml(text: str, name: str, attributes: dict[str, str]) -> str:
    """Rewrites `pos`/`quat`/`size` on the live `<geom name=...>` element.

    Args:
        text: MJCF file contents.
        name: Geom name.
        attributes: Attribute name to new value; missing ones are appended.

    Raises:
        KeyError: No uncommented element with that name.
        ValueError: The element uses `fromto`, which can't be patched safely.
    """
    comments = [m.span() for m in re.finditer(r'<!--.*?-->', text, re.S)]
    pattern = re.compile(rf'<geom\b[^>]*?\bname="{re.escape(name)}"[^>]*?/>', re.S)
    for match in pattern.finditer(text):
        if any(lo <= match.start() < hi for lo, hi in comments):
            continue
        element = match.group(0)
        if re.search(r'\sfromto\s*=', element):
            raise ValueError(f'{name} uses fromto; convert it to pos/quat first.')
        for attr, value in attributes.items():
            existing = re.compile(rf'(\s){attr}="[^"]*"')
            if existing.search(element):
                element = existing.sub(lambda m: f'{m.group(1)}{attr}="{value}"', element, count=1)
            else:
                element = element[:-2].rstrip() + f' {attr}="{value}"/>'
        return text[:match.start()] + element + text[match.end():]
    raise KeyError(f'No live <geom name="{name}"> in the file.')


def write_modified(tuner: CollisionTuner, xml_paths: list[Path]) -> list[str]:
    """Patches each edited geom into the file that defines it.

    A composed model spreads geoms across files (e.g. torso in the Spot file,
    arm in the arm file), so every path is searched. Files that change get a
    `.bak` copy. All edits are applied in memory first, so a failure writes
    nothing.

    Returns:
        Names of the geoms written.

    Raises:
        KeyError: A geom isn't defined in any of `xml_paths`.
    """
    texts = {path: path.read_text() for path in xml_paths}
    written = []
    for geom_id in tuner.modified_ids():
        name = tuner.name(geom_id)
        attributes = geom_attributes(tuner.model, geom_id)
        for path, text in texts.items():
            try:
                texts[path] = patch_geom_xml(text, name, attributes)
            except KeyError:
                continue
            written.append(name)
            break
        else:
            raise KeyError(f'{name} is not defined in any of: {", ".join(p.name for p in xml_paths)}')
    for path, text in texts.items():
        if text != path.read_text():
            shutil.copyfile(path, path.with_suffix(path.suffix + '.bak'))
            path.write_text(text)
    if written:
        tuner.commit()
    return written


# Collision meshes already replaced by primitives in spot_simplified.xml,
# paired with their body; drawn translucent as fitting references.
_REFERENCE_MESHES = (
    ('body_collision', 'body'),
    ('arm_link_sh0_base_coll', 'arm_link_sh0'),
    ('arm_link_sh0_left_motor_coll', 'arm_link_sh0'),
    ('arm_link_sh0_right_motor_coll', 'arm_link_sh0'),
    ('arm_link_hr0_coll', 'arm_link_hr0'),
    ('arm_link_el0_coll', 'arm_link_el0'),
    ('arm_link_el1_main_coll', 'arm_link_el1'),
    ('arm_link_el1_lip_coll', 'arm_link_el1'),
    ('arm_link_wr0_coll', 'arm_link_wr0'),
    ('arm_link_wr1_coll', 'arm_link_wr1'),
    ('front_jaw_coll', 'arm_link_wr1'),
    ('middle_jaw_coll', 'arm_link_wr1'),
    ('jaw_tooth_coll', 'arm_link_wr1'),
    ('left_hinge_coll', 'arm_link_fngr'),
    ('left_finger_coll', 'arm_link_fngr'),
    ('left_tooth_coll', 'arm_link_fngr'),
    ('right_hinge_coll', 'arm_link_fngr'),
    ('right_finger_coll', 'arm_link_fngr'),
    ('right_tooth_coll', 'arm_link_fngr'),
)
_TUNABLE_RGBA = np.array([0.2, 0.5, 1.0, 0.55], np.float32)
_SELECTED_RGBA = np.array([1.0, 0.6, 0.1, 0.8], np.float32)
_REFERENCE_RGBA = np.array([1.0, 0.2, 0.2, 0.25], np.float32)


def find_reference_meshes(model: mujoco.MjModel) -> list[tuple[int, int]]:
    """Returns (mesh_id, body_id) for reference meshes no geom uses anymore."""
    used = {int(model.geom_dataid[g]) for g in range(model.ngeom)
            if int(model.geom_type[g]) == int(_GEOM.mjGEOM_MESH)}
    found = []
    for mesh_name, body_name in _REFERENCE_MESHES:
        mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, mesh_name)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if mesh_id >= 0 and body_id >= 0 and mesh_id not in used:
            found.append((mesh_id, body_id))
    return found


class TunerViewer:
    """GLFW window that renders the model and routes input to a tuner."""

    HELP = (
        'Tab/Shift+Tab select | Backspace reset | click select\n'
        'Arrows, PgUp/PgDn: move parent-frame X,Y,Z (1mm; Shift 5mm)\n'
        'Q/A W/S E/D: size[0,1,2] +/- | R/F T/G Y/H: rotate X,Y,Z (5deg; Shift 1deg)\n'
        'Shift+left-drag move, Shift+right-drag rotate | V visuals, M ref meshes, C other collision\n'
        'P print | Ctrl+S save to XML | Esc quit'
    )

    def __init__(self, model, data, tuner, reference, xml_paths: list[Path], visible: bool = True):
        if not glfw.init():
            raise RuntimeError('GLFW failed to initialize.')
        glfw.window_hint(glfw.VISIBLE, visible)
        self.window = glfw.create_window(1400, 900, 'Spot collision tuner', None, None)
        if not self.window:
            raise RuntimeError('GLFW could not create a window.')
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.model, self.data, self.tuner = model, data, tuner
        self.reference, self.xml_paths = reference, xml_paths
        self.show_reference = True
        self.message = ''
        self.opt, self.pert, self.cam = mujoco.MjvOption(), mujoco.MjvPerturb(), mujoco.MjvCamera()
        self.scn = mujoco.MjvScene(model, maxgeom=2000)
        self.con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.opt.geomgroup[:] = [1, 1, 1, 1, 1, 0]
        mujoco.mjv_defaultFreeCamera(model, self.cam)
        self.cam.lookat[:] = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'body')]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.8, 135.0, -25.0
        self._buttons = {glfw.MOUSE_BUTTON_LEFT: False, glfw.MOUSE_BUTTON_RIGHT: False}
        self._last, self._press = (0.0, 0.0), None
        self._restyle(initial=True)
        self._bind_callbacks()

    def _restyle(self, initial: bool = False) -> None:
        model, tuner = self.model, self.tuner
        if initial:
            for g in range(model.ngeom):
                if model.geom_group[g] == 3:
                    model.geom_rgba[g] = _REFERENCE_RGBA
            for g in tuner.geom_ids:
                model.geom_group[g] = TUNE_GROUP
        for g in tuner.geom_ids:
            model.geom_rgba[g] = _SELECTED_RGBA if g == tuner.geom_id else _TUNABLE_RGBA

    def _add_reference_meshes(self) -> None:
        if not self.show_reference:
            return
        model, data, scn = self.model, self.data, self.scn
        for mesh_id, body_id in self.reference:
            if scn.ngeom >= scn.maxgeom:
                return
            body_rot = data.xmat[body_id].reshape(3, 3)
            mesh_rot = np.zeros(9)
            mujoco.mju_quat2Mat(mesh_rot, model.mesh_quat[mesh_id])
            pos = data.xpos[body_id] + body_rot @ model.mesh_pos[mesh_id]
            mat = (body_rot @ mesh_rot.reshape(3, 3)).flatten()
            geom = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(geom, _GEOM.mjGEOM_MESH, np.zeros(3), pos, mat, _REFERENCE_RGBA)
            geom.dataid = 2 * mesh_id
            scn.ngeom += 1

    def report(self, modified_only: bool = True) -> str:
        """Returns MJCF attribute strings for edited (or all) geoms."""
        ids = self.tuner.modified_ids() if modified_only else self.tuner.geom_ids
        lines = []
        for g in ids:
            attrs = geom_attributes(self.model, g)
            lines.append(f'{self.tuner.name(g)}: ' + ' '.join(f'{k}="{v}"' for k, v in attrs.items()))
        return '\n'.join(lines) or '(no edits)'

    def _overlay_text(self) -> tuple[str, str]:
        tuner, g = self.tuner, self.tuner.geom_id
        attrs = geom_attributes(self.model, g)
        kind = mujoco.mjtGeom(int(self.model.geom_type[g])).name.removeprefix('mjGEOM_').lower()
        star = '  (edited)' if tuner.is_modified(g) else ''
        left = f'[{tuner.index + 1}/{len(tuner.geom_ids)}] {tuner.name(g)}{star}\ntype\npos\nquat\nsize\n{self.message}'
        right = f'\n{kind}\n{attrs["pos"]}\n{attrs["quat"]}\n{attrs["size"]}\n'
        return left, right

    def render(self) -> mujoco.MjrRect:
        """Draws one frame into the current framebuffer and returns its viewport."""
        width, height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(self.model, self.data, self.opt, self.pert, self.cam,
                               int(mujoco.mjtCatBit.mjCAT_ALL), self.scn)
        self._add_reference_meshes()
        mujoco.mjr_render(viewport, self.scn, self.con)
        left, right = self._overlay_text()
        font = int(mujoco.mjtFont.mjFONT_NORMAL)
        mujoco.mjr_overlay(font, int(mujoco.mjtGridPos.mjGRID_TOPLEFT), viewport, left, right, self.con)
        mujoco.mjr_overlay(font, int(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT), viewport, self.HELP, '', self.con)
        return viewport

    _NUDGE = {
        glfw.KEY_RIGHT: (0, 1), glfw.KEY_LEFT: (0, -1),
        glfw.KEY_UP: (1, 1), glfw.KEY_DOWN: (1, -1),
        glfw.KEY_PAGE_UP: (2, 1), glfw.KEY_PAGE_DOWN: (2, -1),
    }
    _SIZE = {
        glfw.KEY_Q: (0, 1), glfw.KEY_A: (0, -1), glfw.KEY_W: (1, 1),
        glfw.KEY_S: (1, -1), glfw.KEY_E: (2, 1), glfw.KEY_D: (2, -1),
    }
    _ROTATE = {
        glfw.KEY_R: (0, 1), glfw.KEY_F: (0, -1), glfw.KEY_T: (1, 1),
        glfw.KEY_G: (1, -1), glfw.KEY_Y: (2, 1), glfw.KEY_H: (2, -1),
    }

    def _bind_callbacks(self) -> None:
        glfw.set_mouse_button_callback(self.window, self._on_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        glfw.set_key_callback(self.window, self._on_key)

    def _pixel_scale(self) -> float:
        return glfw.get_framebuffer_size(self.window)[0] / glfw.get_window_size(self.window)[0]

    def _on_button(self, window, button, action, mods) -> None:
        x, y = glfw.get_cursor_pos(window)
        if button in self._buttons:
            self._buttons[button] = action == glfw.PRESS
        self._last = (x, y)
        if button != glfw.MOUSE_BUTTON_LEFT:
            return
        if action == glfw.PRESS:
            self._press = (x, y, time.monotonic())
        elif self._press is not None:
            px, py, t0 = self._press
            if np.hypot(x - px, y - py) < 4 and time.monotonic() - t0 < 0.4:
                self._pick(x, y)
            self._press = None

    def _on_cursor(self, window, x, y) -> None:
        dx, dy = x - self._last[0], y - self._last[1]
        self._last = (x, y)
        left, right = self._buttons[glfw.MOUSE_BUTTON_LEFT], self._buttons[glfw.MOUSE_BUTTON_RIGHT]
        if not (left or right):
            return
        shift = glfw.PRESS in (glfw.get_key(window, glfw.KEY_LEFT_SHIFT), glfw.get_key(window, glfw.KEY_RIGHT_SHIFT))
        scale = self._pixel_scale()
        height = glfw.get_framebuffer_size(window)[1]
        camera = self.scn.camera[0]
        if shift and left:
            self.tuner.drag_in_view(dx * scale, dy * scale, height, camera)
        elif shift and right:
            self.tuner.rotate_in_view(dx * scale, dy * scale, camera)
        else:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V if left else mujoco.mjtMouse.mjMOUSE_MOVE_V
            win_height = glfw.get_window_size(window)[1]
            mujoco.mjv_moveCamera(self.model, int(action), dx / win_height, dy / win_height, self.cam)

    def _on_scroll(self, window, xoffset, yoffset) -> None:
        mujoco.mjv_moveCamera(self.model, int(mujoco.mjtMouse.mjMOUSE_ZOOM), 0.0, -0.05 * yoffset, self.cam)

    def _pick(self, x: float, y: float) -> None:
        width, height = glfw.get_window_size(self.window)
        saved = self.opt.geomgroup.copy()
        self.opt.geomgroup[:] = 0
        self.opt.geomgroup[TUNE_GROUP] = 1
        selpnt = np.zeros(3)
        geom_id, flex_id, skin_id = (np.full(1, -1, np.int32) for _ in range(3))
        mujoco.mjv_select(self.model, self.data, self.opt, width / height,
                          x / width, (height - y) / height, self.scn, selpnt, geom_id, flex_id, skin_id)
        self.opt.geomgroup[:] = saved
        if geom_id[0] >= 0 and self.tuner.select(int(geom_id[0])):
            self._restyle()

    def _save(self) -> None:
        try:
            names = write_modified(self.tuner, self.xml_paths)
        except (KeyError, ValueError, OSError) as error:
            self.message = f'SAVE FAILED: {error}'
        else:
            self.message = f'saved {len(names)} geom(s)'
        print(self.message)

    def _on_key(self, window, key, scancode, action, mods) -> None:
        if action == glfw.RELEASE:
            return
        shift, ctrl = bool(mods & glfw.MOD_SHIFT), bool(mods & glfw.MOD_CONTROL)
        step, turn = (0.005, np.radians(1.0)) if shift else (0.001, np.radians(5.0))
        tuner = self.tuner
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_S and ctrl:
            self._save()
        elif key in self._NUDGE:
            tuner.nudge(self._NUDGE[key][0], self._NUDGE[key][1] * step)
        elif key in self._SIZE:
            tuner.resize(self._SIZE[key][0], self._SIZE[key][1] * step)
        elif key in self._ROTATE:
            tuner.rotate(self._ROTATE[key][0], self._ROTATE[key][1] * turn)
        elif key == glfw.KEY_TAB:
            tuner.cycle(-1 if shift else 1)
            self._restyle()
        elif key == glfw.KEY_BACKSPACE:
            tuner.reset()
        elif key == glfw.KEY_V:
            self.opt.geomgroup[2] ^= 1
        elif key == glfw.KEY_C:
            self.opt.geomgroup[3] ^= 1
        elif key == glfw.KEY_M:
            self.show_reference = not self.show_reference
        elif key == glfw.KEY_P:
            print(self.report(modified_only=False))

    def run(self) -> None:
        """Runs the render/event loop until the window closes."""
        while not glfw.window_should_close(self.window):
            self.render()
            glfw.swap_buffers(self.window)
            glfw.poll_events()
        if self.tuner.modified_ids():
            print('Unsaved edits:\n' + self.report())
        glfw.terminate()


def load_model(scene: Path, pose: str = 'zero') -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Loads `scene` and poses it for viewing (kinematics only).

    Args:
        scene: Scene MJCF.
        pose: `zero` puts every joint at 0 (the base keeps its default
            pose); `home` uses the `home` keyframe.
    """
    # Absolute: MuJoCo joins a relative main path onto <attach>ed model paths twice.
    model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    if pose == 'home':
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
        if key < 0:
            raise ValueError(f'{scene.name} has no `home` keyframe.')
        mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    return model, data


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Interactively fit Spot collision primitives.')
    parser.add_argument('--scene', type=Path, default=_DEFAULT_SCENE)
    parser.add_argument('--xml', type=Path, nargs='+', default=list(_DEFAULT_XMLS),
                        help='MJCF files that Ctrl+S may patch; each geom is written to the '
                             'file that defines it (a .bak copy is kept).')
    parser.add_argument('--geoms', default=None,
                        help='Regex selecting geoms to tune (default: group-3 primitives except feet).')
    parser.add_argument('--pose', choices=('zero', 'home'), default='zero',
                        help='zero: all joints 0 (default); home: stowed-arm keyframe.')
    args = parser.parse_args(argv)
    model, data = load_model(args.scene, args.pose)
    tuner = CollisionTuner(model, data, find_tunable_geoms(model, args.geoms))
    TunerViewer(model, data, tuner, find_reference_meshes(model), args.xml).run()


if __name__ == '__main__':
    main()
