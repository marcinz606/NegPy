import struct
from PyQt6.QtWidgets import QVBoxLayout, QWidget
from PyQt6.QtCore import QTimer
from rendercanvas.pyqt6 import RenderCanvas
import wgpu  # type: ignore
from typing import Optional, Any, Tuple
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)


class GPUCanvasWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.canvas = RenderCanvas(parent=self)
        self.layout().addWidget(self.canvas)

        self.device: Optional[Any] = None
        self.context: Optional[Any] = None
        self.render_pipeline: Optional[Any] = None
        self.current_texture_view: Optional[Any] = None
        self.uniform_buffer: Optional[Any] = None
        self.image_size: Tuple[int, int] = (1, 1)
        self.format: str = ""

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

        self.uniform_buffer = self.device.create_buffer(size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._create_render_pipeline(self.format)

    def update_texture(self, tex_wrapper: Any) -> None:
        self.current_texture_view = tex_wrapper.view
        self.image_size = (tex_wrapper.width, tex_wrapper.height)
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
        struct RenderUniforms { rect: vec4<f32> };
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
            var out: VertexOutput;
            out.pos = vec4<f32>(
                (ndc_pos.x + 1.0) * 0.5 * params.rect.z + params.rect.x,
                (ndc_pos.y - 1.0) * 0.5 * params.rect.w + params.rect.y,
                0.0, 1.0
            );
            out.uv = uvs[in_vertex_index];
            return out;
        }

        @group(0) @binding(0) var tex: texture_2d<f32>;

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
            return textureSampleBicubic(in.uv);
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
                        "min_binding_size": 16,
                    },
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
        except Exception:
            # Swapchain might be invalid during rapid resizing
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
                    "clear_value": (0.06, 0.06, 0.06, 1),
                }
            ]
        )

        if self.current_texture_view:
            ww, wh = float(current_tex.width), float(current_tex.height)
            iw, ih = float(self.image_size[0]), float(self.image_size[1])

            r = min(ww / iw, wh / ih)
            nw, nh = iw * r, ih * r
            nx, ny = (ww - nw) / 2.0, (wh - nh) / 2.0

            self.device.queue.write_buffer(
                self.uniform_buffer,
                0,
                struct.pack(
                    "ffff",
                    (nx / ww) * 2.0 - 1.0,
                    1.0 - (ny / wh) * 2.0,
                    (nw / ww) * 2.0,
                    (nh / wh) * 2.0,
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
                            "size": 16,
                        },
                    },
                ],
            )

            pass_enc.set_pipeline(self.render_pipeline)
            pass_enc.set_bind_group(0, bind_group)
            pass_enc.draw(4, 1, 0, 0)

        pass_enc.end()
        self.device.queue.submit([enc.finish()])
