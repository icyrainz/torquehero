import json

from torquehero.config import Config, config_dir


def test_defaults_match_spec():
    c = Config()
    assert (c.perfect_window, c.good_window) == (0.05, 0.12)
    assert (c.perfect_pos, c.good_pos, c.road_tolerance) == (0.08, 0.20, 0.20)
    assert (c.expr_perfect, c.expr_good) == (0.15, 0.30)
    assert (c.spin_perfect_deg, c.spin_good_deg) == (360.0, 240.0)
    assert (c.score_perfect, c.score_good, c.combo_step, c.max_multiplier) == (100, 50, 10, 4)
    assert c.ffb_gain == 0.5
    assert c.echo_max_deg_s == 180.0
    assert c.play_range_deg == 90.0


def test_save_load_roundtrip(tmp_path):
    c = Config(ffb_gain=0.3, audio_offset=0.012)
    p = c.save(tmp_path / "c.json")
    assert Config.load(p) == c


def test_load_missing_gives_defaults(tmp_path):
    assert Config.load(tmp_path / "nope.json") == Config()


def test_load_toml_and_ignore_unknown(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('ffb_gain = 0.7\nold_key = 1\n')
    assert Config.load(p).ffb_gain == 0.7


def test_ffb_gain_never_above_one(tmp_path):
    from torquehero.config import settings_fallback

    assert Config(ffb_gain=3.0).ffb_gain == 1.0
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"ffb_gain": 3.0}))
    cfg = Config.load(p)
    assert cfg.ffb_gain == Config().ffb_gain and settings_fallback(cfg)   # a bad value, not full gain


def test_config_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("TORQUEHERO_CONFIG_DIR", str(tmp_path))
    assert config_dir() == tmp_path
    Config(ffb_gain=0.4).save()
    assert Config.load().ffb_gain == 0.4


def test_ffb_gain_clamped_on_construction():
    assert Config(ffb_gain=2.0).ffb_gain == 1.0 and Config(ffb_gain=-1).ffb_gain == 0.0


def test_playability_defaults():
    c = Config()
    assert c.limb_travel == 0.4
    assert c.wheel_range_deg == 1080.0  # SPEC 9 rule 12: one turn plus the play range stays clear of the lock
    assert c.max_lane_rate == {"easy": 1.0, "normal": 1.5, "hard": 2.5}


def test_max_lane_rate_roundtrip(tmp_path):
    c = Config(max_lane_rate={"easy": 0.5, "normal": 1.0, "hard": 2.0})
    assert Config.load(c.save(tmp_path / "c.json")) == c


def test_spin_tolerance_default():
    assert Config().spin_tolerance_deg == 15.0


def test_partial_max_lane_rate_merges_defaults(tmp_path):
    c = Config.from_dict({"max_lane_rate": {"normal": 2.0}})
    assert c.max_lane_rate == {"easy": 1.0, "normal": 2.0, "hard": 2.5}
    p = tmp_path / "c.toml"
    p.write_text("[max_lane_rate]\nhard = 3.0\n")
    assert Config.load(p).max_lane_rate == {"easy": 1.0, "normal": 1.5, "hard": 3.0}


def test_spin_tolerance_clamped():
    assert Config(spin_tolerance_deg=-5).spin_tolerance_deg == 0.0
    assert Config(spin_tolerance_deg=500).spin_tolerance_deg == 120.0   # spin_perfect_deg - spin_good_deg
    assert Config(spin_perfect_deg=300, spin_good_deg=290, spin_tolerance_deg=15).spin_tolerance_deg == 10.0


def test_corrupt_settings_fall_back_key_by_key_with_one_warning(tmp_path, caplog):
    from torquehero.audio import AudioSettings
    from torquehero.config import settings_fallback
    from torquehero.render import RenderSettings

    cases = [
        (Config, "{not json", Config()),
        (Config, "[1, 2]", Config()),
        (Config, '{"play_range_deg": "wide", "ffb_gain": 0.15}', Config(ffb_gain=0.15)),
        (Config, '{"max_lane_rate": {"easy": "fast"}, "audio_offset": NaN}', Config()),
        (Config, '{"ffb_gain": 15, "master_volume": 0.6}', Config(master_volume=0.6)),   # 15 meant as 15%
        (Config, '{"ffb_gain": -0.1}', Config()),
        (Config, '{"lookahead": 1' + "0" * 400 + '}', Config()),                       # huge int: no crash
        (AudioSettings, '{"blocksize": 1' + "0" * 400 + '}', AudioSettings()),
        (AudioSettings, '{"latency": 1' + "0" * 400 + '}', AudioSettings()),
        (RenderSettings, '{"fps": 1' + "0" * 400 + '}', RenderSettings()),
        (AudioSettings, "{", AudioSettings()),
        (AudioSettings, '{"blocksize": "512", "device": 3}', AudioSettings(device=3)),
        (AudioSettings, '{"device": [1], "samplerate": 44100.5, "latency": 0.02}', AudioSettings(latency=0.02)),
        (RenderSettings, "{", RenderSettings()),
        (RenderSettings, '{"vsync": "yes", "fps": 90, "font": 5}', RenderSettings(fps=90)),
    ]
    for cls, bad, want in cases:
        p = tmp_path / "s.json"
        p.write_text(bad)
        caplog.clear()
        got = cls.load(p)
        assert got == want and settings_fallback(got), (cls, bad)
        assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1, (cls, bad)
    p.write_text('{"blocksize": 512, "device": null}')
    assert not settings_fallback(AudioSettings.load(p))


def test_a_settings_file_that_fell_back_is_backed_up_before_the_first_save(tmp_path):
    from torquehero.audio import AudioSettings

    p = tmp_path / "audio.json"
    p.write_text('{"blocksize": "512"}')
    AudioSettings.load(p).save(p)
    assert (tmp_path / "audio.json.bad").read_text() == '{"blocksize": "512"}'
    assert json.loads(p.read_text())["blocksize"] == 256


def test_the_fell_back_marker_stays_until_the_backup_succeeds(tmp_path, monkeypatch):
    from torquehero import config as config_mod

    p = tmp_path / "config.json"
    p.write_text('{"ffb_gain": 15}')
    cfg = Config.load(p)

    def fail(*a):
        raise OSError("disk full")

    monkeypatch.setattr(config_mod.shutil, "copyfile", fail)
    try:
        cfg.save(p)
    except OSError:
        pass
    assert p.read_text() == '{"ffb_gain": 15}'                 # nothing overwritten
    monkeypatch.undo()
    cfg.save(p)
    assert (tmp_path / "config.json.bad").read_text() == '{"ffb_gain": 15}'
