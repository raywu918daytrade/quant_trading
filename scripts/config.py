"""模型角色設定 (2026-10-23 定案: 正式 = 基準+支撐壓力, 影子 = 基準).
MAIN_VARIANT 的產出不帶後綴 (signals/<date>.*, db/signals/latest.json, paper/, db/paper/);
SHADOW_VARIANT 的產出帶 _<variant> 後綴 (signals/<date>_base.*, latest_base.json, paper_base/, db/paper_base/).
要對調角色只改這兩行, 並重新部署 (workflow 與 web/mobile.html 的 SHADOW 常數要一致)."""
MAIN_VARIANT = "sr"
SHADOW_VARIANT = "base"
VARIANTS = ["base", "sr", "seq"]
def suffix(variant: str) -> str:
    return "" if variant == MAIN_VARIANT else "_" + variant
