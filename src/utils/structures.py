import math
import numpy as np
from enum import IntEnum, unique
from typing import List, Tuple, Union, Any, Dict, BinaryIO
import warnings
import itertools
import torch
from torch.nn import functional as F
import pickle
from torch import device

_RawBoxType = Union[List[float], Tuple[float, ...], torch.Tensor, np.ndarray]

@unique
class BoxMode(IntEnum):
    """
    Enum of different ways to represent a box.
    """

    XYXY_ABS = 0
    """
    (x0, y0, x1, y1) in absolute floating points coordinates.
    The coordinates in range [0, width or height].
    """
    XYWH_ABS = 1
    """
    (x0, y0, w, h) in absolute floating points coordinates.
    """
    XYXY_REL = 2
    """
    Not yet supported!
    (x0, y0, x1, y1) in range [0, 1]. They are relative to the size of the image.
    """
    XYWH_REL = 3
    """
    Not yet supported!
    (x0, y0, w, h) in range [0, 1]. They are relative to the size of the image.
    """
    XYWHA_ABS = 4
    """
    (xc, yc, w, h, a) in absolute floating points coordinates.
    (xc, yc) is the center of the rotated box, and the angle a is in degrees ccw.
    """

    @staticmethod
    def convert(box: _RawBoxType, from_mode: "BoxMode", to_mode: "BoxMode") -> _RawBoxType:
        """
        Args:
            box: can be a k-tuple, k-list or an Nxk array/tensor, where k = 4 or 5
            from_mode, to_mode (BoxMode)

        Returns:
            The converted box of the same type.
        """
        if from_mode == to_mode:
            return box

        original_type = type(box)
        is_numpy = isinstance(box, np.ndarray)
        single_box = isinstance(box, (list, tuple))
        if single_box:
            assert len(box) == 4 or len(box) == 5, (
                "BoxMode.convert takes either a k-tuple/list or an Nxk array/tensor,"
                " where k == 4 or 5"
            )
            arr = torch.tensor(box)[None, :]
        else:
            # avoid modifying the input box
            if is_numpy:
                arr = torch.from_numpy(np.asarray(box)).clone()
            else:
                arr = box.clone()

        assert to_mode not in [BoxMode.XYXY_REL, BoxMode.XYWH_REL] and from_mode not in [
            BoxMode.XYXY_REL,
            BoxMode.XYWH_REL,
        ], "Relative mode not yet supported!"

        if from_mode == BoxMode.XYWHA_ABS and to_mode == BoxMode.XYXY_ABS:
            assert (
                arr.shape[-1] == 5
            ), "The last dimension of input shape must be 5 for XYWHA format"
            original_dtype = arr.dtype
            arr = arr.double()

            w = arr[:, 2]
            h = arr[:, 3]
            a = arr[:, 4]
            c = torch.abs(torch.cos(a * math.pi / 180.0))
            s = torch.abs(torch.sin(a * math.pi / 180.0))
            # This basically computes the horizontal bounding rectangle of the rotated box
            new_w = c * w + s * h
            new_h = c * h + s * w

            # convert center to top-left corner
            arr[:, 0] -= new_w / 2.0
            arr[:, 1] -= new_h / 2.0
            # bottom-right corner
            arr[:, 2] = arr[:, 0] + new_w
            arr[:, 3] = arr[:, 1] + new_h

            arr = arr[:, :4].to(dtype=original_dtype)
        elif from_mode == BoxMode.XYWH_ABS and to_mode == BoxMode.XYWHA_ABS:
            original_dtype = arr.dtype
            arr = arr.double()
            arr[:, 0] += arr[:, 2] / 2.0
            arr[:, 1] += arr[:, 3] / 2.0
            angles = torch.zeros((arr.shape[0], 1), dtype=arr.dtype)
            arr = torch.cat((arr, angles), axis=1).to(dtype=original_dtype)
        else:
            if to_mode == BoxMode.XYXY_ABS and from_mode == BoxMode.XYWH_ABS:
                arr[:, 2] += arr[:, 0]
                arr[:, 3] += arr[:, 1]
            elif from_mode == BoxMode.XYXY_ABS and to_mode == BoxMode.XYWH_ABS:
                arr[:, 2] -= arr[:, 0]
                arr[:, 3] -= arr[:, 1]
            else:
                raise NotImplementedError(
                    "Conversion from BoxMode {} to {} is not supported yet".format(
                        from_mode, to_mode
                    )
                )

        if single_box:
            return original_type(arr.flatten().tolist())
        if is_numpy:
            return arr.numpy()
        else:
            return arr

class Boxes:
    """
    This structure stores a list of boxes as a Nx4 torch.Tensor.
    It supports some common methods about boxes
    (`area`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)

    Attributes:
        tensor (torch.Tensor): float matrix of Nx4. Each row is (x1, y1, x2, y2).
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx4 matrix.  Each row is (x1, y1, x2, y2).
        """
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor, dtype=torch.float32, device=torch.device("cpu"))
        else:
            tensor = tensor.to(torch.float32)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, 4)).to(dtype=torch.float32)
        assert tensor.dim() == 2 and tensor.size(-1) == 4, tensor.size()

        self.tensor = tensor

    def clone(self) -> "Boxes":
        """
        Clone the Boxes.

        Returns:
            Boxes
        """
        return Boxes(self.tensor.clone())

    def to(self, device: torch.device):
        # Boxes are assumed float32 and does not support to(dtype)
        return Boxes(self.tensor.to(device=device))

    def area(self) -> torch.Tensor:
        """
        Computes the area of all the boxes.

        Returns:
            torch.Tensor: a vector with areas of each box.
        """
        box = self.tensor
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        return area

    def clip(self, box_size: Tuple[int, int]) -> None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].

        Args:
            box_size (height, width): The clipping box's size.
        """
        assert torch.isfinite(self.tensor).all(), "Box tensor contains infinite or NaN!"
        h, w = box_size
        x1 = self.tensor[:, 0].clamp(min=0, max=w)
        y1 = self.tensor[:, 1].clamp(min=0, max=h)
        x2 = self.tensor[:, 2].clamp(min=0, max=w)
        y2 = self.tensor[:, 3].clamp(min=0, max=h)
        self.tensor = torch.stack((x1, y1, x2, y2), dim=-1)

    def nonempty(self, threshold: float = 0.0) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        box = self.tensor
        widths = box[:, 2] - box[:, 0]
        heights = box[:, 3] - box[:, 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def __getitem__(self, item) -> "Boxes":
        """
        Args:
            item: int, slice, or a BoolTensor

        Returns:
            Boxes: Create a new :class:`Boxes` by indexing.

        The following usage are allowed:

        1. `new_boxes = boxes[3]`: return a `Boxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned Boxes might share storage with this Boxes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Boxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return Boxes(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "Boxes(" + str(self.tensor) + ")"

    def inside_box(self, box_size: Tuple[int, int], boundary_threshold: int = 0) -> torch.Tensor:
        """
        Args:
            box_size (height, width): Size of the reference box.
            boundary_threshold (int): Boxes that extend beyond the reference box
                boundary by more than boundary_threshold are considered "outside".

        Returns:
            a binary vector, indicating whether each box is inside the reference box.
        """
        height, width = box_size
        inds_inside = (
            (self.tensor[..., 0] >= -boundary_threshold)
            & (self.tensor[..., 1] >= -boundary_threshold)
            & (self.tensor[..., 2] < width + boundary_threshold)
            & (self.tensor[..., 3] < height + boundary_threshold)
        )
        return inds_inside

    def get_centers(self) -> torch.Tensor:
        """
        Returns:
            The box centers in a Nx2 array of (x, y).
        """
        return (self.tensor[:, :2] + self.tensor[:, 2:]) / 2

    def scale(self, scale_x: float, scale_y: float) -> None:
        """
        Scale the box with horizontal and vertical scaling factors
        """
        self.tensor[:, 0::2] *= scale_x
        self.tensor[:, 1::2] *= scale_y

    @classmethod
    def cat(cls, boxes_list: List["Boxes"]) -> "Boxes":
        """
        Concatenates a list of Boxes into a single Boxes

        Arguments:
            boxes_list (list[Boxes])

        Returns:
            Boxes: the concatenated Boxes
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all([isinstance(box, Boxes) for box in boxes_list])

        # use torch.cat (v.s. layers.cat) so the returned boxes never share storage with input
        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    @property
    def device(self) -> device:
        return self.tensor.device

    # type "Iterator[torch.Tensor]", yield, and iter() not supported by torchscript
    # https://github.com/pytorch/pytorch/issues/18627
    @torch.jit.unused
    def __iter__(self):
        """
        Yield a box as a Tensor of shape (4,) at a time.
        """
        yield from self.tensor

class Instances:
    """
    This class represents a list of instances in an image.
    It stores the attributes of instances (e.g., boxes, masks, labels, scores) as "fields".
    All fields must have the same ``__len__`` which is the number of instances.

    All other (non-field) attributes of this class are considered private:
    they must start with '_' and are not modifiable by a user.

    Some basic usage:

    1. Set/get/check a field:

       .. code-block:: python

          instances.gt_boxes = Boxes(...)
          print(instances.pred_masks)  # a tensor of shape (N, H, W)
          print('gt_masks' in instances)

    2. ``len(instances)`` returns the number of instances
    3. Indexing: ``instances[indices]`` will apply the indexing on all the fields
       and returns a new :class:`Instances`.
       Typically, ``indices`` is a integer vector of indices,
       or a binary mask of length ``num_instances``

       .. code-block:: python

          category_3_detections = instances[instances.pred_classes == 3]
          confident_detections = instances[instances.scores > 0.9]
    """

    def __init__(self, image_size: Tuple[int, int], **kwargs: Any):
        """
        Args:
            image_size (height, width): the spatial size of the image.
            kwargs: fields to add to this `Instances`.
        """
        self._image_size = image_size
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self) -> Tuple[int, int]:
        """
        Returns:
            tuple: height, width
        """
        return self._image_size

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        with warnings.catch_warnings(record=True):
            data_len = len(value)
        if len(self._fields):
            assert (
                len(self) == data_len
            ), "Adding a field of length {} to a Instances of length {}".format(data_len, len(self))
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        return self._fields

    # Tensor-like methods
    def to(self, *args: Any, **kwargs: Any) -> "Instances":
        """
        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)
        return ret

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instances":
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            # use __len__ because len() has to be int and is not friendly to tracing
            return v.__len__()
        raise NotImplementedError("Empty Instances does not support __len__!")

    def __iter__(self):
        raise NotImplementedError("`Instances` object is not iterable!")

    @staticmethod
    def cat(instance_lists: List["Instances"]) -> "Instances":
        """
        Args:
            instance_lists (list[Instances])

        Returns:
            Instances
        """
        assert all(isinstance(i, Instances) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        image_size = instance_lists[0].image_size
        if not isinstance(image_size, torch.Tensor):  # could be a tensor in tracing
            for i in instance_lists[1:]:
                assert i.image_size == image_size
        ret = Instances(image_size)
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)
        return ret

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "image_height={}, ".format(self._image_size[0])
        s += "image_width={}, ".format(self._image_size[1])
        s += "fields=[{}])".format(", ".join((f"{k}: {v}" for k, v in self._fields.items())))
        return s

    __repr__ = __str__

class TorchSerializedList(object):
    """
    A list-like object whose items are serialized and stored in a torch tensor. When
    launching a process that uses TorchSerializedList with "fork" start method,
    the subprocess can read the same buffer without triggering copy-on-access. When
    launching a process that uses TorchSerializedList with "spawn/forkserver" start
    method, the list will be pickled by a special ForkingPickler registered by PyTorch
    that moves data to shared memory. In both cases, this allows parent and child
    processes to share RAM for the list data, hence avoids the issue in
    https://github.com/pytorch/pytorch/issues/13246.

    See also https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/
    on how it works.
    """

    def __init__(self, lst: list):
        self._lst = lst

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        print(
            "Serializing {} elements to byte tensors and concatenating them all ...".format(
                len(self._lst)
            )
        )
        self._lst = [_serialize(x) for x in self._lst]
        self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
        self._addr = torch.from_numpy(np.cumsum(self._addr))
        self._lst = torch.from_numpy(np.concatenate(self._lst))
        print("Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024**2))

    def __len__(self):
        return len(self._addr)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr].numpy())

        # @lint-ignore PYTHONPICKLEISBAD
        return pickle.loads(bytes)

# from densepose.data.meshes.catalog import MeshCatalog
# from densepose.structures.mesh import load_mesh_symmetry
# from densepose.structures.transform_data import DensePoseTransformData

class DensePoseTransformData(object):

    # Horizontal symmetry label transforms used for horizontal flip
    MASK_LABEL_SYMMETRIES = [0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14]
    # fmt: off
    POINT_LABEL_SYMMETRIES = [ 0, 1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23]  # noqa
    # fmt: on

    def __init__(self, uv_symmetries: Dict[str, torch.Tensor], device: torch.device):
        self.mask_label_symmetries = DensePoseTransformData.MASK_LABEL_SYMMETRIES
        self.point_label_symmetries = DensePoseTransformData.POINT_LABEL_SYMMETRIES
        self.uv_symmetries = uv_symmetries
        self.device = torch.device("cpu")

    def to(self, device: torch.device, copy: bool = False) -> "DensePoseTransformData":
        """
        Convert transform data to the specified device

        Args:
            device (torch.device): device to convert the data to
            copy (bool): flag that specifies whether to copy or to reference the data
                in case the device is the same
        Return:
            An instance of `DensePoseTransformData` with data stored on the specified device
        """
        if self.device == device and not copy:
            return self
        uv_symmetry_map = {}
        for key in self.uv_symmetries:
            uv_symmetry_map[key] = self.uv_symmetries[key].to(device=device, copy=copy)
        return DensePoseTransformData(uv_symmetry_map, device)

    @staticmethod
    def load(io: Union[str, BinaryIO]):
        """
        Args:
            io: (str or binary file-like object): input file to load data from
        Returns:
            An instance of `DensePoseTransformData` with transforms loaded from the file
        """
        import scipy.io

        uv_symmetry_map = scipy.io.loadmat(io)
        uv_symmetry_map_torch = {}
        for key in ["U_transforms", "V_transforms"]:
            uv_symmetry_map_torch[key] = []
            map_src = uv_symmetry_map[key]
            map_dst = uv_symmetry_map_torch[key]
            for i in range(map_src.shape[1]):
                map_dst.append(torch.from_numpy(map_src[0, i]).to(dtype=torch.float))
            uv_symmetry_map_torch[key] = torch.stack(map_dst, dim=0)
        transform_data = DensePoseTransformData(uv_symmetry_map_torch, device=torch.device("cpu"))
        return transform_data

class DensePoseDataRelative(object):
    """
    Dense pose relative annotations that can be applied to any bounding box:
        x - normalized X coordinates [0, 255] of annotated points
        y - normalized Y coordinates [0, 255] of annotated points
        i - body part labels 0,...,24 for annotated points
        u - body part U coordinates [0, 1] for annotated points
        v - body part V coordinates [0, 1] for annotated points
        segm - 256x256 segmentation mask with values 0,...,14
    To obtain absolute x and y data wrt some bounding box one needs to first
    divide the data by 256, multiply by the respective bounding box size
    and add bounding box offset:
        x_img = x0 + x_norm * w / 256.0
        y_img = y0 + y_norm * h / 256.0
    Segmentation masks are typically sampled to get image-based masks.
    """

    # Key for normalized X coordinates in annotation dict
    X_KEY = "dp_x"
    # Key for normalized Y coordinates in annotation dict
    Y_KEY = "dp_y"
    # Key for U part coordinates in annotation dict (used in chart-based annotations)
    U_KEY = "dp_U"
    # Key for V part coordinates in annotation dict (used in chart-based annotations)
    V_KEY = "dp_V"
    # Key for I point labels in annotation dict (used in chart-based annotations)
    I_KEY = "dp_I"
    # Key for segmentation mask in annotation dict
    S_KEY = "dp_masks"
    # # Key for vertex ids (used in continuous surface embeddings annotations)
    # VERTEX_IDS_KEY = "dp_vertex"
    # # Key for mesh id (used in continuous surface embeddings annotations)
    # MESH_NAME_KEY = "ref_model"
    # Number of body parts in segmentation masks
    N_BODY_PARTS = 14
    # Number of parts in point labels
    N_PART_LABELS = 24
    MASK_SIZE = 256

    def __init__(self, annotation, cleanup=False):
        self.x = torch.as_tensor(annotation[DensePoseDataRelative.X_KEY])
        self.y = torch.as_tensor(annotation[DensePoseDataRelative.Y_KEY])
        if (
            DensePoseDataRelative.I_KEY in annotation
            and DensePoseDataRelative.U_KEY in annotation
            and DensePoseDataRelative.V_KEY in annotation
        ):
            self.i = torch.as_tensor(annotation[DensePoseDataRelative.I_KEY])
            self.u = torch.as_tensor(annotation[DensePoseDataRelative.U_KEY])
            self.v = torch.as_tensor(annotation[DensePoseDataRelative.V_KEY])
        # if (
        #     DensePoseDataRelative.VERTEX_IDS_KEY in annotation
        #     and DensePoseDataRelative.MESH_NAME_KEY in annotation
        # ):
        #     self.vertex_ids = torch.as_tensor(
        #         annotation[DensePoseDataRelative.VERTEX_IDS_KEY], dtype=torch.long
        #     )
        #     # self.mesh_id = MeshCatalog.get_mesh_id(annotation[DensePoseDataRelative.MESH_NAME_KEY])
        if DensePoseDataRelative.S_KEY in annotation:
            self.segm = DensePoseDataRelative.extract_segmentation_mask(annotation)
        self.device = torch.device("cpu")
        if cleanup:
            DensePoseDataRelative.cleanup_annotation(annotation)

    def to(self, device):
        if self.device == device:
            return self
        new_data = DensePoseDataRelative.__new__(DensePoseDataRelative)
        new_data.x = self.x.to(device)
        new_data.y = self.y.to(device)
        for attr in ["i", "u", "v", "vertex_ids", "segm"]:
            if hasattr(self, attr):
                setattr(new_data, attr, getattr(self, attr).to(device))
        # if hasattr(self, "mesh_id"):
        #     new_data.mesh_id = self.mesh_id
        new_data.device = device
        return new_data

    @staticmethod
    def extract_segmentation_mask(annotation):
        import pycocotools.mask as mask_utils

        # TODO: annotation instance is accepted if it contains either
        # DensePose segmentation or instance segmentation. However, here we
        # only rely on DensePose segmentation
        poly_specs = annotation[DensePoseDataRelative.S_KEY]
        if isinstance(poly_specs, torch.Tensor):
            # data is already given as mask tensors, no need to decode
            return poly_specs
        segm = torch.zeros((DensePoseDataRelative.MASK_SIZE,) * 2, dtype=torch.float32)
        if isinstance(poly_specs, dict):
            if poly_specs:
                mask = mask_utils.decode(poly_specs)
                segm[mask > 0] = 1
        else:
            for i in range(len(poly_specs)):
                poly_i = poly_specs[i]
                if poly_i:
                    mask_i = mask_utils.decode(poly_i)
                    segm[mask_i > 0] = i + 1
        return segm

    @staticmethod
    def validate_annotation(annotation):
        for key in [
            DensePoseDataRelative.X_KEY,
            DensePoseDataRelative.Y_KEY,
        ]:
            if key not in annotation:
                return False, "no {key} data in the annotation".format(key=key)
        valid_for_iuv_setting = all(
            key in annotation
            for key in [
                DensePoseDataRelative.I_KEY,
                DensePoseDataRelative.U_KEY,
                DensePoseDataRelative.V_KEY,
            ]
        )
        # valid_for_cse_setting = all(
        #     key in annotation
        #     for key in [
        #         DensePoseDataRelative.VERTEX_IDS_KEY,
        #         DensePoseDataRelative.MESH_NAME_KEY,
        #     ]
        # )
        # if not valid_for_iuv_setting and not valid_for_cse_setting:
        if not valid_for_iuv_setting:
            return (
                False,
                "expected either {} (IUV setting) annotations".format(
                    ", ".join(
                        [
                            DensePoseDataRelative.I_KEY,
                            DensePoseDataRelative.U_KEY,
                            DensePoseDataRelative.V_KEY,
                        ]
                    ),
                    # ", ".join(
                    #     [
                    #         DensePoseDataRelative.VERTEX_IDS_KEY,
                    #         DensePoseDataRelative.MESH_NAME_KEY,
                    #     ]
                    # ),
                ),
            )
        return True, None

    @staticmethod
    def cleanup_annotation(annotation):
        for key in [
            DensePoseDataRelative.X_KEY,
            DensePoseDataRelative.Y_KEY,
            DensePoseDataRelative.I_KEY,
            DensePoseDataRelative.U_KEY,
            DensePoseDataRelative.V_KEY,
            DensePoseDataRelative.S_KEY,
            # DensePoseDataRelative.VERTEX_IDS_KEY,
            # DensePoseDataRelative.MESH_NAME_KEY,
        ]:
            if key in annotation:
                del annotation[key]

    def apply_transform(self, transforms, densepose_transform_data):
        self._transform_pts(transforms, densepose_transform_data)
        if hasattr(self, "segm"):
            self._transform_segm(transforms, densepose_transform_data)

    def _transform_pts(self, transforms, dp_transform_data):
        import transform as T

        # NOTE: This assumes that HorizFlipTransform is the only one that does flip
        do_hflip = sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1
        if do_hflip:
            self.x = self.MASK_SIZE - self.x
            if hasattr(self, "i"):
                self._flip_iuv_semantics(dp_transform_data)
            if hasattr(self, "vertex_ids"):
                self._flip_vertices()

        for t in transforms.transforms:
            if isinstance(t, T.RotationTransform):
                xy_scale = np.array((t.w, t.h)) / DensePoseDataRelative.MASK_SIZE
                xy = t.apply_coords(np.stack((self.x, self.y), axis=1) * xy_scale)
                self.x, self.y = torch.tensor(xy / xy_scale, dtype=self.x.dtype).T

    def _flip_iuv_semantics(self, dp_transform_data: DensePoseTransformData) -> None:
        i_old = self.i.clone()
        uv_symmetries = dp_transform_data.uv_symmetries
        pt_label_symmetries = dp_transform_data.point_label_symmetries
        for i in range(self.N_PART_LABELS):
            if i + 1 in i_old:
                annot_indices_i = i_old == i + 1
                if pt_label_symmetries[i + 1] != i + 1:
                    self.i[annot_indices_i] = pt_label_symmetries[i + 1]
                u_loc = (self.u[annot_indices_i] * 255).long()
                v_loc = (self.v[annot_indices_i] * 255).long()
                self.u[annot_indices_i] = uv_symmetries["U_transforms"][i][v_loc, u_loc].to(
                    device=self.u.device
                )
                self.v[annot_indices_i] = uv_symmetries["V_transforms"][i][v_loc, u_loc].to(
                    device=self.v.device
                )

    # def _flip_vertices(self):
    #     mesh_info = MeshCatalog[MeshCatalog.get_mesh_name(self.mesh_id)]
    #     mesh_symmetry = (
    #         load_mesh_symmetry(mesh_info.symmetry) if mesh_info.symmetry is not None else None
    #     )
    #     self.vertex_ids = mesh_symmetry["vertex_transforms"][self.vertex_ids]

    def _transform_segm(self, transforms, dp_transform_data):
        import transform as T

        # NOTE: This assumes that HorizFlipTransform is the only one that does flip
        do_hflip = sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1
        if do_hflip:
            self.segm = torch.flip(self.segm, [1])
            self._flip_segm_semantics(dp_transform_data)

        for t in transforms.transforms:
            if isinstance(t, T.RotationTransform):
                self._transform_segm_rotation(t)

    def _flip_segm_semantics(self, dp_transform_data):
        old_segm = self.segm.clone()
        mask_label_symmetries = dp_transform_data.mask_label_symmetries
        for i in range(self.N_BODY_PARTS):
            if mask_label_symmetries[i + 1] != i + 1:
                self.segm[old_segm == i + 1] = mask_label_symmetries[i + 1]

    def _transform_segm_rotation(self, rotation):
        self.segm = F.interpolate(self.segm[None, None, :], (rotation.h, rotation.w)).numpy()
        self.segm = torch.tensor(rotation.apply_segmentation(self.segm[0, 0]))[None, None, :]
        self.segm = F.interpolate(self.segm, [DensePoseDataRelative.MASK_SIZE] * 2)[0, 0]

class DensePoseList(object):

    _TORCH_DEVICE_CPU = torch.device("cpu")

    def __init__(self, densepose_datas, boxes_xyxy_abs, image_size_hw, device=_TORCH_DEVICE_CPU):
        assert len(densepose_datas) == len(
            boxes_xyxy_abs
        ), "Attempt to initialize DensePoseList with {} DensePose datas " "and {} boxes".format(
            len(densepose_datas), len(boxes_xyxy_abs)
        )
        self.densepose_datas = []
        for densepose_data in densepose_datas:
            assert isinstance(densepose_data, DensePoseDataRelative) or densepose_data is None, (
                "Attempt to initialize DensePoseList with DensePose datas "
                "of type {}, expected DensePoseDataRelative".format(type(densepose_data))
            )
            densepose_data_ondevice = (
                densepose_data.to(device) if densepose_data is not None else None
            )
            self.densepose_datas.append(densepose_data_ondevice)
        self.boxes_xyxy_abs = boxes_xyxy_abs.to(device)
        self.image_size_hw = image_size_hw
        self.device = device

    def to(self, device):
        if self.device == device:
            return self
        return DensePoseList(self.densepose_datas, self.boxes_xyxy_abs, self.image_size_hw, device)

    def __iter__(self):
        return iter(self.densepose_datas)

    def __len__(self):
        return len(self.densepose_datas)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self.densepose_datas))
        s += "image_width={}, ".format(self.image_size_hw[1])
        s += "image_height={})".format(self.image_size_hw[0])
        return s

    def __getitem__(self, item):
        if isinstance(item, int):
            densepose_data_rel = self.densepose_datas[item]
            return densepose_data_rel
        elif isinstance(item, slice):
            densepose_datas_rel = self.densepose_datas[item]
            boxes_xyxy_abs = self.boxes_xyxy_abs[item]
            return DensePoseList(
                densepose_datas_rel, boxes_xyxy_abs, self.image_size_hw, self.device
            )
        elif isinstance(item, torch.Tensor) and (item.dtype == torch.bool):
            densepose_datas_rel = [self.densepose_datas[i] for i, x in enumerate(item) if x > 0]
            boxes_xyxy_abs = self.boxes_xyxy_abs[item]
            return DensePoseList(
                densepose_datas_rel, boxes_xyxy_abs, self.image_size_hw, self.device
            )
        else:
            densepose_datas_rel = [self.densepose_datas[i] for i in item]
            boxes_xyxy_abs = self.boxes_xyxy_abs[item]
            return DensePoseList(
                densepose_datas_rel, boxes_xyxy_abs, self.image_size_hw, self.device
            )