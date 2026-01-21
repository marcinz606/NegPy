import struct
from PyQt6.QtWidgets import QVBoxLayout, QWidget
from rendercanvas.pyqt6 import RenderCanvas
import wgpu  # type: ignore
from typing import Optional, Any, Tuple
from src.kernel.system.logging import get_logger

logger = get_logger(__name__)


class GPUCanvasWidget(QWidget):
    """
    Qt Widget that wraps RenderCanvas for GPU rendering with aspect ratio control.
    Uses a manual bilinear sampler for Rgba32Float compatibility and non-sRGB surface.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        # Create RenderCanvas
        self.canvas = RenderCanvas(parent=self)
        self.layout().addWidget(self.canvas)

        # GPU State
        self.device: Optional[Any] = None
        self.context: Optional[Any] = None
        self.render_pipeline: Optional[Any] = None
        self.current_texture_view: Optional[Any] = None
        self.uniform_buffer: Optional[Any] = None
        self.image_size: Tuple[int, int] = (1, 1)

    def initialize_gpu(self, device: Any, adapter: Any) -> None:
        self.device = device
        self.context = self.canvas.get_context("wgpu")

        preferred_format = self.context.get_preferred_format(adapter)
        # Use non-sRGB format to match CPU's direct-to-screen gamma application
        format = preferred_format.replace("-srgb", "")
        self.context.configure(device=self.device, format=format)

        # Create Uniform Buffer for Quad transform (16 bytes: vec4 rect)
        self.uniform_buffer = self.device.create_buffer(
            size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )

        self._create_render_pipeline(format)

    def update_texture(self, tex_wrapper: Any) -> None:
        """
        Updates the texture to be displayed. Expects a GPUTexture wrapper.
        """
        self.current_texture_view = tex_wrapper.view
        self.image_size = (tex_wrapper.width, tex_wrapper.height)
        self.canvas.request_draw(self._draw_frame)

    def _create_render_pipeline(self, format: str) -> None:
        shader_source = """
        struct RenderUniforms {
            rect: vec4<f32>, // x, y, w, h in NDC
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

        fn textureSampleBilinear(uv: vec2<f32>) -> vec4<f32> {
            let dims = textureDimensions(tex);
            let fdims = vec2<f32>(f32(dims.x), f32(dims.y));
            let pixel = uv * fdims - 0.5;
            let c00 = floor(pixel);
            let t = pixel - c00;
            let i00 = vec2<i32>(c00);
            let idims = vec2<i32>(i32(dims.x), i32(dims.y));
            let v00 = textureLoad(tex, clamp(i00 + vec2<i32>(0, 0), vec2<i32>(0), idims - 1), 0);
            let v10 = textureLoad(tex, clamp(i00 + vec2<i32>(1, 0), vec2<i32>(0), idims - 1), 0);
            let v01 = textureLoad(tex, clamp(i00 + vec2<i32>(0, 1), vec2<i32>(0), idims - 1), 0);
            let v11 = textureLoad(tex, clamp(i00 + vec2<i32>(1, 1), vec2<i32>(0), idims - 1), 0);
            return mix(mix(v00, v10, t.x), mix(v01, v11, t.x), t.y);
        }

        @fragment
        fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
            return textureSampleBilinear(in.uv);
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
            layout=self.device.create_pipeline_layout(
                bind_group_layouts=[self.bind_group_layout]
            ),
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
        if not self.current_texture_view or not self.render_pipeline:
            return
        ww, wh = max(1, self.width()), max(1, self.height())
        iw, ih = self.image_size
        ratio = min(ww / iw, wh / ih)
        nw, nh = iw * ratio, ih * ratio
        nx, ny = (ww - nw) / 2, (wh - nh) / 2
        ndc_x, ndc_y, ndc_w, ndc_h = (
            (nx / ww) * 2 - 1,
            1 - (ny / wh) * 2,
            (nw / ww) * 2,
            (nh / wh) * 2,
        )
        self.device.queue.write_buffer(
            self.uniform_buffer, 0, struct.pack("ffff", ndc_x, ndc_y, ndc_w, ndc_h)
        )
        current_texture = self.context.get_current_texture()
        command_encoder = self.device.create_command_encoder()

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

        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": current_texture.create_view(),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.06, 0.06, 0.06, 1),
                }
            ]
        )

        render_pass.set_pipeline(self.render_pipeline)
        render_pass.set_bind_group(0, bind_group)
        render_pass.draw(4, 1, 0, 0)
        render_pass.end()

        self.device.queue.submit([command_encoder.finish()])
