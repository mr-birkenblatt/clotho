# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete
from torch.onnx import symbolic_helper as symbolic_helper
from torch.onnx import utils as utils
from torch.onnx._globals import GLOBALS as GLOBALS


def hardtanh(g, self, min_val, max_val): ...


def clamp(g, self, min, max): ...


def clamp_min(g, self, min): ...


def clamp_max(g, self, max): ...


def relu6(g, input): ...


def select(g, self, dim, index): ...


def index_put(g, self, indices_list_value, values, accumulate: bool = ...): ...


def pixel_shuffle(g, self, upscale_factor): ...


upsample_nearest1d: Incomplete
upsample_nearest2d: Incomplete
upsample_nearest3d: Incomplete
upsample_linear1d: Incomplete
upsample_bilinear2d: Incomplete
upsample_trilinear3d: Incomplete
upsample_bicubic2d: Incomplete


def gather(g, self, dim, index, sparse_grad: bool = ...): ...


def scatter(g, self, dim, index, src): ...


def cumsum(g, self, dim, dtype: Incomplete | None = ...): ...


def masked_select(g, self, mask): ...


def masked_scatter(g, self, mask, source): ...


def append(g, self, tensor): ...


def add(g, self, other, alpha: Incomplete | None = ...): ...


def insert(g, self, pos, tensor): ...


def pop(g, tensor_list, dim): ...


def Delete(g, tensor_list, dim): ...


def cat(g, tensor_list, dim): ...


def stack(g, tensor_list, dim): ...


avg_pool1d: Incomplete
avg_pool2d: Incomplete
avg_pool3d: Incomplete


def unique_dim(g, self, dim, sorted, return_inverse, return_counts): ...


def topk(g, self, k, dim, largest, sorted, out: Incomplete | None = ...): ...


def sort(g, self, dim, decending, out: Incomplete | None = ...): ...


def round(g, self): ...


def remainder(g, input, other): ...


def split(
    g, self, split_size_or_sizes, dim, _outputs: Incomplete | None = ...): ...


def split_with_sizes(
    g, self, split_sizes, dim, _outputs: Incomplete | None = ...): ...


def unbind(g, self, dim: int = ..., _outputs: Incomplete | None = ...): ...


def constant_pad_nd(g, input, padding, value: Incomplete | None = ...): ...


def reflection_pad(g, input, padding): ...


def replication_pad(g, input, padding): ...


reflection_pad1d = reflection_pad
reflection_pad2d = reflection_pad
reflection_pad3d = reflection_pad
replication_pad1d = replication_pad
replication_pad2d = replication_pad
replication_pad3d = replication_pad


def pad(g, input, pad, mode, value): ...


def linalg_det(g, self): ...


def logdet(g, input): ...


def arange(g, *args): ...


def size(g, self, dim: Incomplete | None = ...): ...


def squeeze(g, self, dim: Incomplete | None = ...): ...


def unsqueeze(g, self, dim): ...


def mm(g, self, other): ...


def index(g, self, index): ...


def index_fill(g, self, dim, index, value): ...


def index_copy(g, self, dim, index, source): ...


def im2col(g, input, kernel_size, dilation, padding, stride): ...


def narrow(g, input, dim, start, length): ...


def flatten(g, input, start_dim, end_dim): ...


def linalg_vector_norm(g, self, ord, dim, keepdim, dtype): ...


def embedding_bag(
    g, embedding_matrix, indices, offsets, scale_grad_by_freq, mode, sparse,
    per_sample_weights, include_last_offset, padding_idx): ...


def embedding_renorm(g, weight, indices, max_norm, norm_type): ...


def chunk(g, self, chunks, dim): ...


def normal(g, loc, scale, seed): ...


class Prim:
    domain: str
    @staticmethod
    def ConstantChunk(g, self, chunks, dim): ...
