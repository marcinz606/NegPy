from negpy.desktop.view.sidebar.demosaic import clip_hint_due


def test_hint_waits_for_a_reading():
    assert not clip_hint_due(None)


def test_hint_needs_a_channel_over_the_warning_share():
    assert not clip_hint_due((0.0, 0.01, 0.0))
    assert clip_hint_due((0.0, 0.012, 0.0))
