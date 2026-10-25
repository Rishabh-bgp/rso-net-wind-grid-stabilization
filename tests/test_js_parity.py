"""JS forward pass (site/model.js) must match the PyTorch predictions stored in site/model.json within 1e-4."""
import json, os, shutil, subprocess, pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_matches_python():
    js = """const fs=require('fs');const W=require('./site/model.js');const M=JSON.parse(fs.readFileSync('site/model.json'));
    console.log(JSON.stringify(M.demo.windows.map(w=>W.forecast(M,w))));"""
    out = json.loads(subprocess.check_output(["node", "-e", js], cwd=ROOT))
    py = json.load(open(os.path.join(ROOT, "site/model.json")))["demo"]["py_pred"]
    assert len(out) == len(py) and max(abs(a - b) for a, b in zip(out, py)) < 1e-4
