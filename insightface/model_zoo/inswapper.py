import time
import numpy as np
import onnxruntime
import cv2
import onnx
import torch
from onnx import numpy_helper
from ..utils import face_align


def hwc_to_bchw(images: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor from HWC format to BCHW format.

    Args:
    images (torch.Tensor): The input tensor in HWC format.

    Returns:
    torch.Tensor: The tensor converted to BCHW format.
    """
    return images.unsqueeze(0).permute(0, 3, 1, 2)


def to_tensor(images: np.ndarray) -> torch.Tensor:
    """
    Convert a numpy array to a PyTorch tensor and change its format from HWC to BCHW.

    Args:
    images (np.ndarray): The input numpy array.

    Returns:
    torch.Tensor: The corresponding PyTorch tensor in BCHW format.
    """
    return hwc_to_bchw(torch.from_numpy(images))


def create_mask(vis_img: np.ndarray) -> np.ndarray:
    """
    Create a binary mask from a given image based on a threshold.

    Args:
    vis_img (np.ndarray): The input image for mask creation.

    Returns:
    np.ndarray: The binary mask of the image.
    """
    threshold = 0.0  # Adjust this threshold as needed
    binary_mask = vis_img > threshold

    # White object on a black background
    object_color, background_color = 255, 0

    white_mask = binary_mask * object_color
    black_mask = ~binary_mask * background_color

    result_mask = white_mask + black_mask
    return result_mask


class INSwapper():
    def __init__(self, model_file=None, session=None):
        self.model_file = model_file
        self.session = session
        model = onnx.load(self.model_file)
        graph = model.graph
        self.emap = numpy_helper.to_array(graph.initializer[-1])
        self.input_mean = 0.0
        self.input_std = 255.0
        #print('input mean and std:', model_file, self.input_mean, self.input_std)
        if self.session is None:
            self.session = onnxruntime.InferenceSession(self.model_file, None)
        inputs = self.session.get_inputs()
        self.input_names = []
        for inp in inputs:
            self.input_names.append(inp.name)
        outputs = self.session.get_outputs()
        output_names = []
        for out in outputs:
            output_names.append(out.name)
        self.output_names = output_names
        assert len(self.output_names)==1
        output_shape = outputs[0].shape
        input_cfg = inputs[0]
        input_shape = input_cfg.shape
        self.input_shape = input_shape
        print('inswapper-shape:', self.input_shape)
        self.input_size = tuple(input_shape[2:4][::-1])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, img, latent):
        img = (img - self.input_mean) / self.input_std
        pred = self.session.run(self.output_names, {self.input_names[0]: img, self.input_names[1]: latent})[0]
        return pred

    

    def get(self, img, target_face, source_face, paste_back=True, dilation=False, static_blur=False):
        aimg, M = face_align.norm_crop2(img, target_face.kps, self.input_size[0])
        blob = cv2.dnn.blobFromImage(aimg, 1.0 / self.input_std, self.input_size,
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        latent = source_face.normed_embedding.reshape((1,-1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        pred = self.session.run(self.output_names, {self.input_names[0]: blob, self.input_names[1]: latent})[0]
        #print(latent.shape, latent.dtype, pred.shape)
        img_fake = pred.transpose((0,2,3,1))[0]
        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:,:,::-1]
        if not paste_back:
            return bgr_fake, M, None
        else:
            target_img = img
            IM = cv2.invertAffineTransform(M)
            aimg_height, aimg_width = aimg.shape[:2]

            # V1
            white_raw = None
            if static_blur:
                white_raw = np.full((aimg_width, aimg_height), 255, dtype=np.float32)
                white_raw = cv2.warpAffine(white_raw, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
                white_raw[white_raw > 20] = 255
                mask_h_inds, mask_w_inds = np.where(white_raw == 255)
                mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
                mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
                mask_size = int(np.sqrt(mask_h * mask_w))

                k = max(mask_size // 10, 10)
                kernel = np.ones((k, k), np.uint8)
                white_raw = cv2.erode(white_raw, kernel, iterations = 1)

                k = max(mask_size // 20, 5)
                kernel_size = (k, k)
                blur_size = tuple(2 * i + 1 for i in kernel_size)
                white_raw = cv2.GaussianBlur(white_raw, blur_size, 0)
                white_raw = white_raw[:, :, np.newaxis]

            #########################################################

            # V2
            bgr_fake = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)

            #########################################################

            # V3
            white_temp = None
            if dilation:
                white_temp = np.full((aimg_width, aimg_height), 0, dtype=np.float32)
                white_temp[int(0.05 * aimg_height) : int(0.95 * aimg_height), int(0.05 * aimg_width) : int(0.95 * aimg_width)] = 1.0
                white_temp = cv2.warpAffine(white_temp, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)

            #########################################################

            # Convert the image to tensor and to the device
            face_image = to_tensor(cv2.cvtColor(bgr_fake, cv2.COLOR_BGR2RGB)).to(device=self.device)

            # Convert the data to PyTorch tensors and move to the device
            boxes_tensor = torch.tensor(np.array([target_face.bbox]), dtype=torch.float32).to(device=self.device)
            key_points_tensor = torch.tensor(np.array([target_face.kps]), dtype=torch.float32).to(device=self.device)

            # Create the dictionary with the desired format
            yv5_faces = {
                'rects': boxes_tensor,
                'points': key_points_tensor,
                'scores': torch.tensor([0.99], dtype=torch.float32).to(device=self.device),
                'image_ids': torch.tensor([0], dtype=torch.int64).to(device=self.device)
            }

            return bgr_fake, white_temp, white_raw, face_image, yv5_faces
