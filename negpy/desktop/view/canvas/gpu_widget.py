import struct
from typing import Any, Optional, Tuple

import numpy as np
import wgpu  # type: ignore
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QVBoxLayout, QWidget
from rendercanvas.pyqt6 import RenderCanvas

from negpy.infrastructure.display.color_mgmt import get_display_lut
from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE
from negpy.infrastructure.display.icc_lut import DEFAULT_LUT_SIZE
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)


class GPUCanvasWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.canvas = RenderCanvas(parent=self)
        self.canvas.setStyleSheet("background-color: #050505;")
        self.layout().addWidget(self.canvas)

        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor("#050505"))
        self.setPalette(pal)
        self.setAutoFillBackground(True)

        self.device: Optional[Any] = None
        self.context: Optional[Any] = None
        self.render_pipeline: Optional[Any] = None
        self.current_texture_view: Optional[Any] = None
        self.uniform_buffer: Optional[Any] = None
        self.image_size: Tuple[int, int] = (1, 1)
        self.format: str = ""

        # Working-space to display-profile LUT (3D texture plus linear sampler). Re-uploaded when
        # the monitor profile changes, for example when the window moves to another screen. None
        # monitor bytes means an sRGB display.
        self.lut_view: Optional[Any] = None
        self.lut_sampler: Optional[Any] = None
        self._monitor_icc_bytes: Optional[bytes] = None
        self._working_color_space: str = WORKING_COLOR_SPACE
        self._proof: Optional[tuple] = None
        # Identity of the LUT currently uploaded, so a settled frame re-uploads nothing.
        self._lut_key: Optional[tuple] = None
        # Grid size of that LUT; the proof and display LUTs do not share one.
        self._lut_n: int = DEFAULT_LUT_SIZE

        self.zoom: float = 1.0
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0
        self._bg: Tuple[float, float, float] = (0.02, 0.02, 0.02)
        self.fit_height_reserve: float = 0.0

        # Debounce resize to prevent context thrashing
        self.resize_timer = QTimer()
        self.resize_timer.setSingleShot(True)
        self.resize_timer.setInterval(50)
        self.resize_timer.timeout.connect(self._perform_resize)

    def _configure_context(self) -> None:
        """
        Helper to configure the WebGPU context, trying available alpha modes
        to handle platform-specific constraints (e.g. 'opaque' vs 'premultiplied').
        """
        if not self.context or not self.device:
            return

        modes = ["premultiplied", "opaque"]
        last_error = None

        for mode in modes:
            try:
                self.context.configure(device=self.device, format=self.format, alpha_mode=mode)
                return
            except Exception as e:
                last_error = e

        if last_error:
            raise last_error

    def initialize_gpu(self, device: Any, adapter: Any) -> None:
        self.device = device
        self.context = self.canvas.get_context("wgpu")

        self.format = self.context.get_preferred_format(adapter).replace("-srgb", "")
        self._configure_context()

        # Uniform buffer: 3 * vec4<f32>
        self.uniform_buffer = self.device.create_buffer(size=48, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._upload_display_lut()
        self._create_render_pipeline(self.format)

        # Initial clear
        self.canvas.request_draw(self._draw_frame)

    def set_monitor_profile(self, monitor_icc_bytes: Optional[bytes]) -> None:
        """Update the display profile and re-upload the working→display LUT.

        Called when the monitor profile is first detected or changes (screen move).
        No-op until the GPU device exists; `initialize_gpu` builds the first LUT.
        """
        if monitor_icc_bytes == self._monitor_icc_bytes:
            return
        self._monitor_icc_bytes = monitor_icc_bytes
        if self.device is not None:
            self._upload_display_lut()
            self._lut_key = (self._working_color_space, monitor_icc_bytes, self._proof)
            self.canvas.request_draw(self._draw_frame)

    def set_display_transform(self, working_color_space: str, monitor_icc_bytes: Optional[bytes], proof: Optional[tuple]) -> None:
        """Point the shader at the LUT for the current display state.

        ``proof`` folds the preview soft proof into the same LUT, so a rendered
        texture reaches the screen without ever coming back to the host.
        """
        key = (working_color_space, monitor_icc_bytes, proof)
        if key == self._lut_key:
            return
        self._lut_key = key
        self._working_color_space = working_color_space
        self._monitor_icc_bytes = monitor_icc_bytes
        self._proof = proof
        if self.device is not None:
            self._upload_display_lut()

    def _upload_display_lut(self) -> None:
        """Build and upload the working-space → display-profile 3D LUT used at
        display time.

        The destination is the monitor profile (`self._monitor_icc_bytes`, or sRGB
        when None), with the soft proof folded in when one is active. On any failure
        an identity LUT is uploaded so the binding always exists and display is a
        pass-through.

        Stored as rgba16float: 8-bit texels would quantize the proof away, and f32 is
        not filterable in core WebGPU.
        """
        try:
            lut = get_display_lut(self._working_color_space, self._monitor_icc_bytes, self._proof)
        except Exception as e:
            logger.warning("Display LUT build failed, using identity: %s", e)
            lut = None

        if lut is None:
            # Identity ramp: texel value == its normalized coordinate.
            n = DEFAULT_LUT_SIZE
            axis = np.linspace(0.0, 1.0, n, dtype=np.float32)
            r, g, b = np.meshgrid(axis, axis, axis, indexing="ij")
            lut = np.stack((r, g, b), axis=-1).astype(np.float32)
        n = lut.shape[0]
        self._lut_n = n

        # Reorder to (b, g, r, c) so the fastest data axis (r) maps to texture width.
        arr = np.ascontiguousarray(lut.transpose(2, 1, 0, 3))
        rgba = np.ones((n, n, n, 4), dtype=np.float16)
        rgba[..., :3] = arr.astype(np.float16)
        rgba = np.ascontiguousarray(rgba)

        lut_tex = self.device.create_texture(
            size=(n, n, n),
            dimension=wgpu.TextureDimension.d3,
            format=wgpu.TextureFormat.rgba16float,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        self.device.queue.write_texture(
            {"texture": lut_tex},
            rgba,
            {"bytes_per_row": n * 8, "rows_per_image": n},
            (n, n, n),
        )
        self.lut_view = lut_tex.create_view(dimension=wgpu.TextureViewDimension.d3)
        self.lut_sampler = self.device.create_sampler(
            mag_filter=wgpu.FilterMode.linear,
            min_filter=wgpu.FilterMode.linear,
            address_mode_u=wgpu.AddressMode.clamp_to_edge,
            address_mode_v=wgpu.AddressMode.clamp_to_edge,
            address_mode_w=wgpu.AddressMode.clamp_to_edge,
        )

    def presents_to_screen(self) -> bool:
        """True when the frame goes to a native surface of its own, which the overlay
        must punch an alpha hole to reveal.

        rendercanvas picks a *bitmap* present on Qt: the frame is blitted into the
        shared backing store by the widget's own paintEvent, and a hole punched over
        it erases the frame instead of revealing anything.
        """
        sub = getattr(self.canvas, "_subwidget", self.canvas)
        return bool(getattr(sub, "_present_to_screen", False))

    def set_transform(self, zoom: float, px: float, py: float) -> None:
        self.zoom = zoom
        self.pan_x = px
        self.pan_y = py
        self.canvas.request_draw(self._draw_frame)

    def update_texture(self, tex_wrapper: Any) -> None:
        self.current_texture_view = tex_wrapper.view
        self.image_size = (tex_wrapper.width, tex_wrapper.height)
        self.canvas.request_draw(self._draw_frame)

    def set_background_color(self, r: float, g: float, b: float) -> None:
        self._bg = (r, g, b)
        hex_color = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        self.canvas.setStyleSheet(f"background-color: {hex_color};")
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor(hex_color))
        self.setPalette(pal)
        self.canvas.request_draw(self._draw_frame)

    def clear(self) -> None:
        self.current_texture_view = None
        self.canvas.request_draw(self._draw_frame)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.resize_timer.start()

    def _perform_resize(self) -> None:
        if self.device and self.context:
            try:
                self._configure_context()

                if self.current_texture_view:
                    self.canvas.request_draw(self._draw_frame)
            except Exception as e:
                logger.error(f"Failed to reconfigure WebGPU context on resize: {e}")

    def _create_render_pipeline(self, format: str) -> None:
        shader_source = """
        struct RenderUniforms {
            rect: vec4<f32>,
            transform: vec4<f32>, // x: zoom, y: pan_x, z: pan_y, w: LUT grid size
            filt: vec4<f32>       // x: image pixels per screen pixel
        };
        @group(0) @binding(1) var<uniform> params: RenderUniforms;

        struct VertexOutput {
            @builtin(position) pos: vec4<f32>,
            @location(0) uv: vec2<f32>,
        };

        @vertex
        fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
            var positions = array<vec2<f32>, 4>(
                vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, 1.0),
                vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0)
            );
            var uvs = array<vec2<f32>, 4>(
                vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0),
                vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0)
            );

            let ndc_pos = positions[in_vertex_index];
            let zoom = params.transform.x;
            let pan = params.transform.yz;

            // 1. Fit to baseline (Centered)
            let base_pos = vec2<f32>(
                (ndc_pos.x + 1.0) * 0.5 * params.rect.z + params.rect.x,
                (ndc_pos.y - 1.0) * 0.5 * params.rect.w + params.rect.y
            );

            // 2. Zoom and Pan relative to center of viewport
            let final_pos = base_pos * zoom + vec2<f32>(pan.x, -pan.y) * 2.0;

            var out: VertexOutput;
            out.pos = vec4<f32>(final_pos, 0.0, 1.0);
            out.uv = uvs[in_vertex_index];
            return out;
        }

        @group(0) @binding(0) var tex: texture_2d<f32>;
        @group(0) @binding(2) var lut_tex: texture_3d<f32>;
        @group(0) @binding(3) var lut_samp: sampler;

        // Working-space → display-profile LUT grid size, uploaded per LUT: the soft
        // proof uses a finer grid than the plain display transform.
        fn lut_coord(v: f32, n: f32) -> f32 {
            // Map a [0,1] value to the texture coord at the matching texel center.
            return (0.5 + clamp(v, 0.0, 1.0) * (n - 1.0)) / n;
        }

        fn cubic(v: f32) -> f32 {
            let a = 0.5;
            let x = abs(v);
            if (x < 1.0) {
                return 1.5 * x * x * x - 2.5 * x * x + 1.0;
            } else if (x < 2.0) {
                return -0.5 * x * x * x + 2.5 * x * x - 4.0 * x + 2.0;
            }
            return 0.0;
        }

        // Minification path. Bicubic reads a 4x4 window of a much denser grid, so it
        // is no more correct than bilinear there and costs four times the taps.
        fn textureSampleBilinear(uv: vec2<f32>) -> vec4<f32> {
            let dims = textureDimensions(tex);
            let fdims = vec2<f32>(f32(dims.x), f32(dims.y));
            let pixel = uv * fdims - 0.5;
            let ipos = floor(pixel);
            let fpos = fract(pixel);

            var col = vec4<f32>(0.0);
            for (var y = 0; y <= 1; y++) {
                for (var x = 0; x <= 1; x++) {
                    let coord = clamp(vec2<i32>(ipos + vec2<f32>(f32(x), f32(y))), vec2<i32>(0), vec2<i32>(dims) - 1);
                    let wx = select(1.0 - fpos.x, fpos.x, x == 1);
                    let wy = select(1.0 - fpos.y, fpos.y, y == 1);
                    col += textureLoad(tex, coord, 0) * (wx * wy);
                }
            }
            return col;
        }

        fn textureSampleBicubic(uv: vec2<f32>) -> vec4<f32> {
            let dims = textureDimensions(tex);
            let fdims = vec2<f32>(f32(dims.x), f32(dims.y));

            let pixel = uv * fdims - 0.5;
            let ipos = floor(pixel);
            let fpos = fract(pixel);

            var col = vec4<f32>(0.0);

            for (var y = -1; y <= 2; y++) {
                for (var x = -1; x <= 2; x++) {
                    let offset = vec2<f32>(f32(x), f32(y));
                    let coord = vec2<i32>(ipos + offset);

                    let c = clamp(coord, vec2<i32>(0), vec2<i32>(dims) - 1);

                    let weight = cubic(f32(x) - fpos.x) * cubic(f32(y) - fpos.y);
                    col += textureLoad(tex, c, 0) * weight;
                }
            }
            return col;
        }

        @fragment
        fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
            // filt.x >= 1 = magnifying, the only regime the extra taps buy anything.
            // Must be if/else: assigning one then overwriting runs both.
            var col: vec4<f32>;
            if (params.filt.x >= 1.0) {
                col = textureSampleBicubic(in.uv);
            } else {
                col = textureSampleBilinear(in.uv);
            }
            let n = params.transform.w;
            let coord = vec3<f32>(lut_coord(col.r, n), lut_coord(col.g, n), lut_coord(col.b, n));
            let mapped = textureSampleLevel(lut_tex, lut_samp, coord, 0.0).rgb;
            return vec4<f32>(mapped, col.a);
        }
        """
        shader = self.device.create_shader_module(code=shader_source)
        self.bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.unfilterable_float,
                        "view_dimension": wgpu.TextureViewDimension.d2,
                    },
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {
                        "type": wgpu.BufferBindingType.uniform,
                        "min_binding_size": 48,
                    },
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": wgpu.TextureViewDimension.d3,
                    },
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": wgpu.SamplerBindingType.filtering},
                },
            ]
        )
        self.render_pipeline = self.device.create_render_pipeline(
            layout=self.device.create_pipeline_layout(bind_group_layouts=[self.bind_group_layout]),
            vertex={"module": shader, "entry_point": "vs_main"},
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_strip,
                "strip_index_format": wgpu.IndexFormat.uint32,
            },
            fragment={
                "module": shader,
                "entry_point": "fs_main",
                "targets": [{"format": format}],
            },
        )

    def _draw_frame(self) -> None:
        if not self.render_pipeline or not self.context:
            return

        try:
            current_tex = self.context.get_current_texture()
        except (RuntimeError, wgpu.GPUError) as exc:
            # Swapchain unavailable during resize, so skip the frame
            logger.debug("swapchain unavailable during resize: %s", exc)
            return

        if current_tex is None:
            return

        enc = self.device.create_command_encoder()
        pass_enc = enc.begin_render_pass(
            color_attachments=[
                {
                    "view": current_tex.create_view(),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (*self._bg, 1),
                }
            ]
        )

        if self.current_texture_view:
            ww, wh = float(current_tex.width), float(current_tex.height)
            iw, ih = float(self.image_size[0]), float(self.image_size[1])

            dpr = self.devicePixelRatioF()
            reserve_px = self.fit_height_reserve * dpr
            fit_h = max(1.0, wh - reserve_px)
            r = min(ww / iw, fit_h / ih)
            nw, nh = iw * r, ih * r

            nx = (ww - nw) / 2.0
            ny = (fit_h - nh) / 2.0

            self.device.queue.write_buffer(
                self.uniform_buffer,
                0,
                struct.pack(
                    "ffffffffffff",
                    (nx / ww) * 2.0 - 1.0,
                    1.0 - (ny / wh) * 2.0,
                    (nw / ww) * 2.0,
                    (nh / wh) * 2.0,
                    self.zoom,
                    self.pan_x,
                    self.pan_y,
                    float(self._lut_n),
                    # Screen pixels per image pixel: r fits the image, zoom scales it.
                    r * self.zoom,
                    0.0,
                    0.0,
                    0.0,
                ),
            )

            bind_group = self.device.create_bind_group(
                layout=self.bind_group_layout,
                entries=[
                    {"binding": 0, "resource": self.current_texture_view},
                    {
                        "binding": 1,
                        "resource": {
                            "buffer": self.uniform_buffer,
                            "offset": 0,
                            "size": 48,
                        },
                    },
                    {"binding": 2, "resource": self.lut_view},
                    {"binding": 3, "resource": self.lut_sampler},
                ],
            )

            pass_enc.set_pipeline(self.render_pipeline)
            pass_enc.set_bind_group(0, bind_group)
            pass_enc.draw(4, 1, 0, 0)

        pass_enc.end()
        self.device.queue.submit([enc.finish()])
