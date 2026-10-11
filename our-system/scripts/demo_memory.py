"""Show the local memory core using synthetic messages and the real API handlers."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import uuid
from datetime import datetime,timezone
from fastapi.testclient import TestClient
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.api import create_app
from app.config import Settings


def main():
    out=ROOT/"artifacts"/("demo-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:6])
    out.mkdir(parents=True,exist_ok=False)
    app=create_app(Settings(db_path=out/"memory.sqlite",enable_governance=True,context_neighbor_radius=1))
    transcript=[]
    with TestClient(app) as client:
        def add(user,request,text,time):
            response=client.post("/add",json={"user_id":user,"request_id":request,"session_id":"s",
                                             "messages":[{"role":"user","content":text,"timestamp":time}]})
            if response.status_code!=200: raise RuntimeError("demo Add failed")
        def search(label,user,question):
            result=client.post("/search",json={"user_id":user,"query":question,"top_k":10})
            if result.status_code!=200: raise RuntimeError("demo Search failed")
            transcript.append({"scenario":label,"question":question,"evidence":result.json()["data"]})
        add("residence","old","我住在苏州。",1704067200000)
        add("residence","new","我搬到扬州了。",1735689600000)
        search("明确更新后的当前证据","residence","现在住在什么城市")
        search("保留历史","residence","2024-06-01 住在什么城市")
        add("conflict","one","我的时区是 UTC+8。",1704067200000)
        add("conflict","two","我的时区是 UTC+9。",1735689600000)
        search("没有明确更新，保留冲突双方","conflict","现在的时区记录")
        add("deletion","original","我的代号是 cedar7420。",1704067200000)
        add("deletion","erase",'删除“我的代号是 cedar7420。”',1735689600000)
        add("deletion","original","我的代号是 cedar7420。",1704067200000)
        app.state.store.rebuild_user_lexical_index("deletion")
        search("删除后重放与重建","deletion","cedar7420")
        transcript.append({"scenario":"内部删除回执","receipts":app.state.store.lifecycle_receipts("deletion","erase")})
        add("withdrawal","one","我对榛子过敏。",1704067200000)
        add("withdrawal","two",'撤回我之前说的“我对榛子过敏。”',1735689600000)
        search("撤回后不作当前支持","withdrawal","目前的过敏记录")
        search("明确历史用途带撤回归因","withdrawal","以前撤回的过敏记录")
    report={"scope":"synthetic local API demonstration, not a benchmark","external_model_calls":0,
            "default_settings_changed":False,"transcript":transcript}
    (out/"result.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print("Artifacts: "+str(out))
if __name__=="__main__":
    main()
