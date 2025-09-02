@group(0) @binding(0) var ids: texture_2d<u32>;
@group(0) @binding(1) var out_color: texture_storage_2d<rgba8unorm, write>;

struct Colors { rgba: array<vec4<f32>> };
@group(0) @binding(3) var<storage, read> colors: Colors;

@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(out_color);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let id = textureLoad(ids, vec2<i32>(gid.xy), 0).r;
  var rgba: vec4<f32>;
  if (id == 0xfffffffFu) {
    rgba = vec4<f32>(0.0, 0.0, 0.0, 1.0);
  } else {
    rgba = colors.rgba[id];
  }
  textureStore(out_color, vec2<i32>(gid.xy), rgba);
}
