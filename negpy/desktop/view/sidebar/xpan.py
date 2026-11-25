from negpy.kernel.system.i18n import tr

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import wrap_tooltip
from negpy.infrastructure.scanners.roll import XPAN_MODE_SETTING

_TOGGLE_TIP = tr("Let each frame in the Coolscan strip preview move up to 55 mm, for XPAN panoramas.")


class XpanSidebar(BaseSidebar):
    """XPAN Mode: an app-wide setting that the strip preview reads when it opens."""

    def _init_ui(self) -> None:
        self.enable_btn = self._small_toggle(
            "mdi.panorama-horizontal",
            tr("XPAN Mode"),
            self._saved(),
            wrap_tooltip(_TOGGLE_TIP),
            align_left=True,
        )
        self.layout.addWidget(self.enable_btn)

    def _connect_signals(self) -> None:
        self.enable_btn.toggled.connect(self._on_toggled)

    def _saved(self) -> bool:
        return bool(self.controller.session.repo.get_global_setting(XPAN_MODE_SETTING, False))

    def _on_toggled(self, checked: bool) -> None:
        self.controller.session.repo.save_global_setting(XPAN_MODE_SETTING, bool(checked))

    def sync_ui(self) -> None:
        self.enable_btn.blockSignals(True)
        self.enable_btn.setChecked(self._saved())
        self.enable_btn.blockSignals(False)
