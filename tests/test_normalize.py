from app.normalize import (
    _decode_nato_service_id,
    _hint_call_sign,
    _normalize_handover_officer,
    _normalize_ranks,
    apply_field_normalization,
)


def test_hint_call_sign_lf_a12():
    assert _hint_call_sign("LF-A12 stop for location") == "LF812"


def test_hint_call_sign_lf_digits():
    assert _hint_call_sign("LF812 stop") == "LF812"


def test_normalize_ranks_sergeant_3():
    assert _normalize_ranks("Sergeant 3, Ashraf") == "SGT3, Ashraf"


def test_normalize_ranks_triple_s():
    assert _normalize_ranks("handover to triple S") == "handover to SSS"


def test_decode_nato_service_id():
    assert _decode_nato_service_id("Tango 1, 9-0-3-5-0") == "T190350"


def test_normalize_handover_officer_spoken_rank_and_nato():
    result = _normalize_handover_officer(
        "Sergeant 3, Ashraf. Tango 1, 9-0-3-5-0",
        "",
    )
    assert "SGT3" in result
    assert "T190350" in result


def test_apply_field_normalization_fills_empty_call_sign():
    fields = {"applianceCallSign": "", "handoverOfficer": "", "handoverNpc": ""}
    result = apply_field_normalization(fields, "LF-A12 stop at 7 Gul Ave")
    assert result["applianceCallSign"] == "LF812"


def test_apply_field_normalization_leaves_living_room_unchanged():
    fields = {"areaOfFireOrigin": "living room", "handoverOfficer": "", "handoverNpc": ""}
    result = apply_field_normalization(
        fields,
        "fire originated in the living room at 7 Gul Ave",
    )
    assert result["areaOfFireOrigin"] == "living room"


def test_apply_field_normalization_fills_npc_when_empty():
    fields = {"handoverNpc": "", "handoverOfficer": ""}
    result = apply_field_normalization(fields, "handed over from Nanyang NPC")
    assert result["handoverNpc"] == "Nanyang NPC"
