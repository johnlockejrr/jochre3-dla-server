import os

import cv2
import copy
import numpy as np
import torch
from PIL import Image
from skimage import img_as_bool
from skimage.morphology import convex_hull_image
from skimage.transform import resize
from ultralytics import YOLO

class Analyzer:
    def __init__(self, general_model=None, image_model=None):
        model_path = 'models'
        general_model_name = 'e50_aug.pt'
        image_model_name = 'e100_img.pt'

        if not general_model:
            self.general_model = YOLO(os.path.join(model_path, general_model_name))
        else:
            self.general_model = general_model

        if not image_model:
            self.image_model = YOLO(os.path.join(model_path, image_model_name))
        else:
            self.image_model = image_model

    __configs = {}
    __configs['paratext'] = {
        'sz' : 640,
        'conf': 0.25,
        'rm': True,
        'classes': [0, 1]
    }
    __configs['imgtab'] = {
        'sz' : 640,
        'conf': 0.35,
        'rm': True,
        'classes': [2, 3]
    }
    __configs['image'] = {
        'sz' : 640,
        'conf': 0.35,
        'rm': True,
        'classes': [0]
    }

    __configs_without_retina_mask = {}
    for key in __configs.keys():
        __configs_without_retina_mask[key] = copy.deepcopy(__configs[key])
        __configs_without_retina_mask[key]['rm'] = False

    classes = [
        "paragraph",
        "textbox",
        "image",
        "table"
    ]

    def __tableConvexHull(img, masks):
        mask=np.zeros(masks[0].shape,dtype="bool")
        for msk in masks:
            temp=msk.cpu().detach().numpy();
            chull = convex_hull_image(temp);
            mask=np.bitwise_or(mask,chull)
        return mask

    def __cls_exists(clss, cls):
        indices = torch.where(clss==cls)
        return len(indices[0])>0

    def __empty_mask(img):
        mask = np.zeros(img.shape[:2], dtype="uint8")
        return np.array(mask, dtype=bool)

    def __extract_img_mask(img_model, img, config):
        res_dict = {
            'status' : 1
        }
        res = Analyzer.__get_predictions(img_model, img, config)

        if res['status']==-1:
            res_dict['status'] = -1

        elif res['status']==0:
            res_dict['mask']=Analyzer.__empty_mask(img)

        else:
            masks = res['masks']
            boxes = res['boxes']
            clss = boxes[:, 5]
            mask = Analyzer.__extract_mask(img, masks, boxes, clss, 0)
            res_dict['mask'] = mask
        return res_dict

    def __get_predictions(model, img2, config):
        res_dict = {
            'status': 1
        }
        try:
            for result in model.predict(source=img2, verbose=False, retina_masks=config['rm'],\
                                        imgsz=config['sz'], conf=config['conf'], stream=True,\
                                        classes=config['classes']):
                try:
                    res_dict['masks'] = result.masks.data
                    res_dict['boxes'] = result.boxes.data
                    del result
                    return res_dict
                except Exception as e:
                    res_dict['status'] = 0
                    return res_dict
        except:
            res_dict['status'] = -1
            return res_dict

    def __extract_mask(img, masks, boxes, clss, cls):
        if not Analyzer.__cls_exists(clss, cls):
            return Analyzer.__empty_mask(img)
        indices = torch.where(clss==cls)
        c_masks = masks[indices]
        mask_arr = torch.any(c_masks, dim=0).bool()
        mask_arr = mask_arr.cpu().detach().numpy()
        mask = mask_arr
        return mask


    def __get_masks(self, img, configs):
        response = {
            'status': 1
        }
        ans_masks = []
        img2 = img

        # ***** Getting paragraph and text masks
        res = Analyzer.__get_predictions(self.general_model, img2, configs['paratext'])
        if res['status']==-1:
            response['status'] = -1
            return response
        elif res['status']==0:
            for i in range(2): ans_masks.append(Analyzer.__empty_mask(img))
        else:
            masks, boxes = res['masks'], res['boxes']
            clss = boxes[:, 5]
            for cls in range(2):
                mask = Analyzer.__extract_mask(img, masks, boxes, clss, cls)
                ans_masks.append(mask)


        # ***** Getting image and table masks
        res2 = Analyzer.__get_predictions(self.general_model, img2, configs['imgtab'])
        if res2['status']==-1:
            response['status'] = -1
            return response
        elif res2['status']==0:
            for i in range(2): ans_masks.append(Analyzer.__empty_mask(img))
        else:
            masks, boxes = res2['masks'], res2['boxes']
            clss = boxes[:, 5]

            if Analyzer.__cls_exists(clss, 2):
                img_res = Analyzer.__extract_img_mask(self.image_model, img, configs['image'])
                if img_res['status'] == 1:
                    img_mask = img_res['mask']
                else:
                    response['status'] = -1
                    return response

            else:
                img_mask = Analyzer.__empty_mask(img)
            ans_masks.append(img_mask)

            if Analyzer.__cls_exists(clss, 3):
                indices = torch.where(clss==3)
                tbl_mask = Analyzer.__tableConvexHull(img, masks[indices])
            else:
                tbl_mask = Analyzer.__empty_mask(img)
            ans_masks.append(tbl_mask)

        if not configs['paratext']['rm']:
            h, w, c = img.shape
            for i in range(4):
                ans_masks[i] = img_as_bool(resize(ans_masks[i], (h, w)))


        response['masks'] = ans_masks
        return response

    def __analyze(self, img):
        res = self.__get_masks(img, Analyzer.__configs)
        if res['status']==-1:
            res = self.__get_masks(img, Analyzer.__configs_without_retina_mask)
        return res

    def overlay(image, mask, color, alpha, resize=None):
        """Combines image and its segmentation mask into a single image.
        https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

        Params:
            image: Training image. np.ndarray,
            mask: Segmentation mask. np.ndarray,
            color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
            alpha: Segmentation mask's transparency. float = 0.5,
            resize: If provided, both image and its mask are resized before blending them together.
            tuple[int, int] = (1024, 1024))

        Returns:
            image_combined: The combined image. np.ndarray

        """
        color = color[::-1]
        colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
        colored_mask = np.moveaxis(colored_mask, 0, -1)
        masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
        image_overlay = masked.filled()

        if resize is not None:
            image = cv2.resize(image.transpose(1, 2, 0), resize)
            image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

        image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

        return image_combined

    def __image_to_numpy(self, image: Image):
        image = image.convert('RGB')
        img = np.array(image)
        img = img[:, :, ::-1].copy()
        return img

    def generate_pretty_image(self, image: Image):
        img = self.__image_to_numpy(image)
        res = self.__analyze(img)
        masks = res['masks']

        color_map = {
            0 : (255, 0, 0),
            1 : (0, 255, 0),
            2 : (0, 0, 255),
            3 : (255, 255, 0),
        }
        for i, mask in enumerate(masks):
            img = Analyzer.overlay(image=img, mask=mask, color=color_map[i], alpha=0.4)

        output_image = Image.fromarray(img)
        return output_image

    def generate_masks(self, image: Image):
        img = self.__image_to_numpy(image)
        res = self.__analyze(img)
        masks = res['masks']

        labeled_masks = {}
        for i, mask in enumerate(masks):
            labeled_masks[Analyzer.classes[i]] = mask
        return labeled_masks

    def generate_stacked_mask(self, image: Image) -> Image:
        """
        :param img_path:
        :return: a vertically concatenated png, with the following four masks: paragraph, textbox, image, table
        """
        masks = self.generate_masks(image)
        concatenated_mask = np.concatenate((masks["paragraph"], masks["textbox"], masks["image"], masks["table"]))
        mask_image = Image.fromarray(concatenated_mask)
        return mask_image

def main():
    input_folder = "../input"
    output_folder = "../output"
    for file in sorted(os.listdir(input_folder)):
        input_path = os.path.join(input_folder, file)
        image = Image.open(input_path)
        file_base = os.path.splitext(file)[0]

        analyzer = Analyzer()
        output_image = analyzer.generate_pretty_image(image)

        output_path = os.path.join(output_folder, f"{file_base}_pretty.png")
        output_image.save(output_path)

        masks = analyzer.generate_masks(image)
        for label, mask in masks.items():
            mask_image = Image.fromarray(mask)
            mask_path = f"{file_base}_{label}.png"
            mask_path = os.path.join(output_folder, mask_path)
            mask_image.save(mask_path)

        concatenated_mask = np.concatenate((masks["paragraph"], masks["textbox"], masks["image"], masks["table"]))
        mask_image = Image.fromarray(concatenated_mask)
        mask_path = f"{file_base}_masks.png"
        mask_path = os.path.join(output_folder, mask_path)
        mask_image.save(mask_path)

if __name__ == "__main__":
    main()
