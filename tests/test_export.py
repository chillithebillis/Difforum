"""Camera interchange round trips (AE / Blender / JSON) and the preview endpoint."""

import json

import numpy as np
import pytest

from difforum.core import export as X
from difforum.core.camera import build_camera
from difforum.nodes.routes import compute_preview


def cam3d(n=30):
    vals = {
        "translation_x": [0.8] * n, "translation_y": [-0.3] * n, "translation_z": [-1.2] * n,
        "rotation_3d_x": [0.2] * n, "rotation_3d_y": [0.6] * n, "rotation_3d_z": [0.3] * n,
        "zoom": [1.0] * n,
    }
    return build_camera(vals, n, mode="3d", fov=40.0)


def test_json_round_trip_recovers_the_move():
    c = cam3d()
    data = X.to_json(c, 768, 432, 24.0)
    c2w, focals, w, h, fps = X.from_json(json.dumps(data))
    back = X.world_to_camera_track(c2w, focals, w)
    for a, b in zip(c.deltas[1:], back.deltas[1:]):
        assert np.allclose(np.asarray(a), b, atol=1e-5)


def test_blender_round_trip():
    c = cam3d()
    c2w, focals = X.camera_track_to_world(c, 768, 432)
    frames = []
    for m, fl in zip(c2w, focals):
        mb = X._CV_TO_BL_WORLD @ m @ X._CV_TO_BL_CAM
        mb[:3, 3] *= 0.1
        frames.append({"matrix": mb.tolist(), "focal_px": fl})
    data = {"convention": "blender", "unit_scale": 0.1, "width": 768, "height": 432, "fps": 24,
            "frames": frames}
    c2w_b, *_ = X.from_json(data)
    for a, b in zip(c2w, c2w_b):
        assert np.allclose(a, b, atol=1e-6)


def test_ae_round_trip():
    c = cam3d()
    jsx = X.to_ae_jsx(c, 768, 432, 24.0)
    pos = json.loads(jsx.split("var POS = ")[1].split(";")[0])
    rot = json.loads(jsx.split("var ROT = ")[1].split(";")[0])
    zoom = json.loads(jsx.split("var ZOOM = ")[1].split(";")[0])
    data = {"convention": "ae", "width": 768, "height": 432, "fps": 24, "zoom0": zoom[0],
            "reference_z": X.reference_z(),
            "frames": [{"position": p, "rotation": r, "zoom": z} for p, r, z in zip(pos, rot, zoom)]}
    c2w_ae, *_ = X.from_json(data)
    c2w, _ = X.camera_track_to_world(c, 768, 432)
    for a, b in zip(c2w, c2w_ae):
        assert np.allclose(a, b, atol=1e-3)


@pytest.mark.parametrize("mode", ["2d", "3d"])
def test_scripts_are_valid(mode):
    n = 12
    vals = {"translation_x": [1.5] * n, "rotation_3d_z": [0.4] * n, "zoom": [1.01] * n}
    c = build_camera(vals, n, mode=mode)
    compile(X.to_blender_script(c, 640, 360, 24.0), "blender.py", "exec")
    jsx = X.to_ae_jsx(c, 640, 360, 24.0)
    assert "setValueAtTime" in jsx and jsx.count("{") == jsx.count("}")


def test_2d_export_is_a_nodal_camera():
    n = 10
    c = build_camera({"translation_x": [4.0] * n}, n, mode="2d")
    c2w, focals = X.camera_track_to_world(c, 640, 360)
    assert np.allclose(c2w[-1][:3, 3], 0.0)           # no travel, only rotation
    assert not np.allclose(c2w[-1][:3, :3], np.eye(3))


def test_preview_endpoint_payload():
    from difforum.core.direction import default_timeline
    out = compute_preview({"timeline": default_timeline(96), "frames": 96, "fps": 24,
                           "width": 640, "height": 360})
    assert out["affines"][0][0] == 0 and len(out["affines"][0]) == 7
    assert out["affines"][-1][0] == 95
    assert len(out["strength"]) == 96
