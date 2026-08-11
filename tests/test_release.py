from scripts.verify_release import main


def test_release_contract() -> None:
    assert main() == 0

