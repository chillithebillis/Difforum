// Export the selected After Effects camera layer to Difforum.
// File > Scripts > Run Script File..., with a camera layer selected in the
// active comp. Writes a .json you load with Difforum · Camera Import
// (put it in ComfyUI/input).
//
// Convention: AE pixels (x right, y down, z into the screen), X/Y/Z Rotation
// composed as Rz * Ry * Rx - the same order Camera Export writes. Orientation
// is added to the rotations (exact when one of the two is zero).
(function () {
    var comp = app.project.activeItem;
    if (!(comp && comp instanceof CompItem)) { alert("Open a comp first."); return; }
    var cam = comp.selectedLayers.length ? comp.selectedLayers[0] : null;
    if (!cam || !(cam instanceof CameraLayer)) { alert("Select a camera layer."); return; }

    var fps = comp.frameRate, n = Math.round(comp.workAreaDuration * fps);
    var t0 = comp.workAreaStart;
    var zoom0 = cam.zoom.valueAtTime(t0, false);
    var rows = [];
    function r5(v) { return Math.round(v * 100000) / 100000; }
    for (var i = 0; i < n; i++) {
        var t = t0 + i / fps;
        var p = cam.position.valueAtTime(t, false);
        var o = cam.orientation.valueAtTime(t, false);
        var rx = cam.xRotation.valueAtTime(t, false) + o[0];
        var ry = cam.yRotation.valueAtTime(t, false) + o[1];
        var rz = cam.zRotation.valueAtTime(t, false) + o[2];
        var z = cam.zoom.valueAtTime(t, false);
        rows.push('{"f":' + i + ',"position":[' + r5(p[0]) + ',' + r5(p[1]) + ',' + r5(p[2]) +
                  '],"rotation":[' + r5(rx) + ',' + r5(ry) + ',' + r5(rz) + '],"zoom":' + r5(z) + '}');
    }
    var json = '{"schema":"difforum.camera/1","convention":"ae","fps":' + fps +
               ',"width":' + comp.width + ',"height":' + comp.height + ',"zoom0":' + zoom0 +
               ',"reference_z":50.5,"frames":[' + rows.join(",") + ']}';
    var f = File.saveDialog("Save Difforum camera", "JSON:*.json");
    if (!f) return;
    f.encoding = "UTF-8";
    f.open("w"); f.write(json); f.close();
    alert("Difforum: " + n + " frames written.");
})();
