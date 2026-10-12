"""用户身份和会话归属：只使用临时文件，不访问模型或数据库。"""

import json
from uuid import uuid4

import pytest

from app.identity import SessionIdentity, SessionOwnershipError
from app.agent.storage import delete_session, load_session, save_session


def identity(user_id=None):
    return SessionIdentity(user_id or str(uuid4()), str(uuid4()))


def test_user_and_session_paths_have_separate_scopes(tmp_path):
    a = identity()
    another_window = identity(a.user_id)
    b = identity()
    assert a.session_path(tmp_path) != another_window.session_path(tmp_path)
    assert a.memory_dir(tmp_path) == another_window.memory_dir(tmp_path)
    assert a.session_path(tmp_path).parent != b.session_path(tmp_path).parent
    assert a.session_path(tmp_path).is_relative_to(tmp_path.resolve())


@pytest.mark.parametrize("value", ["../other", "default", "", None, 123])
def test_web_identity_rejects_invalid_ids(value):
    with pytest.raises(ValueError):
        SessionIdentity(value, str(uuid4()))
    with pytest.raises(ValueError):
        SessionIdentity(str(uuid4()), value)


def test_uuid_spelling_is_normalized():
    user_id, session_id = str(uuid4()), str(uuid4())
    assert SessionIdentity(user_id.upper(), session_id.upper()) == SessionIdentity(user_id, session_id)


def test_owned_session_round_trip(tmp_path):
    owner = identity()
    path = owner.session_path(tmp_path)
    messages = [{"role": "user", "content": "A 的问题"}]
    save_session(str(path), messages, None, identity=owner)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert (data["user_id"], data["session_id"]) == (owner.user_id, owner.session_id)
    assert load_session(str(path), identity=owner)["messages"] == messages


@pytest.mark.parametrize("operation", ["load", "save", "delete"])
def test_wrong_owner_cannot_read_overwrite_or_delete(tmp_path, operation):
    owner, stranger = identity(), identity()
    path = owner.session_path(tmp_path)
    save_session(str(path), [{"role": "user", "content": "私有记录"}], None, identity=owner)
    original = path.read_bytes()
    with pytest.raises(SessionOwnershipError):
        if operation == "load":
            load_session(str(path), identity=stranger)
        elif operation == "save":
            save_session(str(path), [], None, identity=stranger)
        else:
            delete_session(str(path), identity=stranger)
    assert path.read_bytes() == original


def test_web_cannot_claim_legacy_file_and_cli_cannot_claim_web_file(tmp_path):
    legacy = tmp_path / "legacy.json"
    save_session(str(legacy), [], None)
    assert load_session(str(legacy)) is not None
    with pytest.raises(SessionOwnershipError):
        load_session(str(legacy), identity=identity())
    owner = identity()
    path = owner.session_path(tmp_path)
    save_session(str(path), [], None, identity=owner)
    with pytest.raises(SessionOwnershipError):
        load_session(str(path))


def test_corrupt_owned_file_is_not_silently_overwritten(tmp_path):
    owner = identity()
    path = owner.session_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("broken json", encoding="utf-8")
    with pytest.raises(SessionOwnershipError):
        save_session(str(path), [], None, identity=owner)
    assert path.read_text(encoding="utf-8") == "broken json"
